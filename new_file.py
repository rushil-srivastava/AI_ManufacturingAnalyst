import time
import os
import sys
import warnings
import json
import re
from typing import Dict, List
from collections import defaultdict
from datetime import datetime

# --- Environment Setup ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Package Imports ---
try:
    import vanna
    from vanna.base import VannaBase
    from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
    import openai
    import mysql.connector
    import pandas as pd
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# --- Configuration ---
db_type = "mysql"
mysql_host = "127.0.0.1"
mysql_port = 3306
mysql_user = "root"
mysql_password = "Raekwon_wtc$36" 
mysql_dbname = "ft_database"
llm_model_name = 'DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf'
local_server_url = 'http://127.0.0.1:8000/v1'
placeholder_api_key = "sk-no-key-required"

# --- Question Classification Constants ---

FEW_SHOT_PROMPT = """
You are a manufacturing domain expert. Classify each question into one of three categories based on these criteria:

DESCRIPTIVE: 
- Questions that ask for direct data or facts
- Usually start with: what, how many, list, show, where, when
- Focus on current or historical state
- Example keywords: count, total, list, find, show, what is, where are

JUDGEMENT:
- Questions that require analysis and evaluation
- Usually start with: why, can, will, should, is it
- Need reasoning about causes or effects
- Example keywords: why, evaluate, assess, analyze, is it possible, what caused

SUGGESTION:
- Questions that ask for recommendations or improvements
- Usually start with: how can, what should, recommend
- Focus on future actions or changes
- Example keywords: improve, optimize, suggest, recommend, how can, what should

Examples:
Q1: "What is the current inventory level?" 
A1: DESCRIPTIVE (asks for current state)

Q2: "Why did we miss our delivery target?"
A2: JUDGEMENT (requires analysis of causes)

Q3: "How can we reduce late deliveries?"
A3: SUGGESTION (asks for improvement recommendations)

Q4: "List all overdue orders"
A4: DESCRIPTIVE (asks for direct data)

Q5: "Should we increase safety stock?"
A5: JUDGEMENT (requires evaluation)

Q6: "What changes should we make to improve efficiency?"
A6: SUGGESTION (asks for improvement recommendations)

Now classify this question:
Q: {question}
A: """

CONTEXT_PROMPTS = {
    "descriptive": "Return ONLY SQL with some explanation with plots.",
    "judgement": "When responding, return the required SQL query, provide a detailed analysis of the data including relevant plots, and offer your Judgement on the findings. Clearly state that it is your Judgement and advise the user to independently verify it.",
    "suggestion": "When responding, return the required SQL query, provide a detailed analysis of the data including relevant plots, and offer your suggestion on the findings. Clearly state that it is your Suggestion and advise the user to independently verify it."
}

# Add a keyword-based classification helper
def keyword_based_classification(question: str) -> str:
    question = question.lower()
    
    descriptive_keywords = ['what is', 'how many', 'list', 'show', 'find', 'where', 'when', 'which', 'count']
    judgement_keywords = ['why', 'could', 'would', 'should', 'evaluate', 'analyze', 'assess', 'is it possible']
    suggestion_keywords = ['how can', 'what should', 'improve', 'recommend', 'suggest', 'advise', 'propose']
    
    # Check for suggestion first (highest priority)
    for keyword in suggestion_keywords:
        if keyword in question:
            return "suggestion"
            
    # Check for judgement second
    for keyword in judgement_keywords:
        if keyword in question:
            return "judgement"
            
    # Default to descriptive
    for keyword in descriptive_keywords:
        if keyword in question:
            return "descriptive"
            
    return None

# Update the classify_question method in MyLocalLlm class

CLASSIFICATION_METRICS = defaultdict(int)

# # --- Pattern-based Classification Function ---
# def pattern_based_classification(question: str) -> str:
#     """Use regex patterns to classify questions."""
#     question = question.lower()
    
#     # Descriptive patterns
#     descriptive_patterns = [
#         r"^what is", r"^how many", r"^list", r"^show", r"^tell me", 
#         r"^find", r"^count", r"^where", r"^when", r"^who"
#     ]
    
#     # Judgment patterns
#     judgment_patterns = [
#         r"^can", r"^will", r"^is it possible", r"^why (was|did|is)", 
#         r"^what caused", r"^is this", r"evaluate", r"assess",r"analyse"
#     ]
    
#     # Advice patterns
#     advice_patterns = [
#         r"^how (do|can|should) (i|we)", r"^what should", r"improve", 
#         r"optimize", r"recommend", r"suggest", r"focus on"
#     ]
    
#     # Check patterns
#     for pattern in descriptive_patterns:
#         if re.search(pattern, question):
#             return "descriptive"
    
#     for pattern in judgment_patterns:
#         if re.search(pattern, question):
#             return "judgment"
    
#     for pattern in advice_patterns:
#         if re.search(pattern, question):
#             return "advice"
    
#     # Default to advice as the most complex type
#     return "advice"


# --- Custom LLM Class with MCP Features ---
class MyLocalLlm(VannaBase):
    def __init__(self, config=None):
        super().__init__(config=config)
        self.model = llm_model_name

    # Implement required abstract methods
    def assistant_message(self, content: str) -> Dict:
        return {"role": "assistant", "content": content}

    def system_message(self, content: str) -> Dict:
        return {"role": "system", "content": content}

    def user_message(self, content: str) -> Dict:
        return {"role": "user", "content": content}
            
    def classify_question(self, question: str) -> str:
        # First try keyword-based classification
        keyword_result = keyword_based_classification(question)
        if keyword_result:
            return keyword_result
            
        # If keyword classification fails, use LLM
        classification_prompt = FEW_SHOT_PROMPT.format(question=question)
        response = self.submit_prompt(classification_prompt)
        
        # Clean up and validate the response
        response = response.strip().lower()
        if response in QUESTION_TYPES:
            return response
        # Default to descriptive if everything fails
        return "descriptive"

    def submit_prompt(self, prompt, **kwargs) -> str:
        try:
            local_client = openai.OpenAI(
                base_url=local_server_url,
                api_key=placeholder_api_key,
                timeout=kwargs.get('request_timeout', 180.0)
            )
            
            messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}]
            
            response = local_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', 0.1),
                max_tokens=kwargs.get('max_tokens', 1536),
                stop=kwargs.get('stop', ["``````"])
            )
            
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            return "Error: No response generated"

        except Exception as e:
            return f"Error: {str(e)}"

# --- Enhanced Vanna Class ---
class MyVanna(ChromaDB_VectorStore, MyLocalLlm):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        MyLocalLlm.__init__(self, config=config)
        
    def generate_contextual_response(self, question: str) -> str:
        category = self.classify_question(question)
        CLASSIFICATION_METRICS[category] += 1
        
        system_msg = f"""You are a manufacturing analyst. Follow these rules:
        - DESCRIPTIVE: {CONTEXT_PROMPTS['descriptive']}
        - JUDGEMENT: {CONTEXT_PROMPTS['judgement']}
        - SUGGESTION: {CONTEXT_PROMPTS['suggestion']}"""
        
        return self.generate_sql(
            question=question,
            chat_history=[self.system_message(system_msg)]
        )

# --- Core Execution Flow --- 
vn = MyVanna(config=None)

# Database connection setup remains unchanged
# --- Connect to Database ---
print(f"Connecting to {db_type} database '{mysql_dbname}' on {mysql_host}...")
try:
    if db_type == "mysql":
        vn.connect_to_mysql(
            host=mysql_host,
            port=mysql_port,
            user=mysql_user,
            password=mysql_password,
            dbname=mysql_dbname
        )
    else:
        print(f"Database type '{db_type}' not configured.")
        sys.exit(1)
    print("Database connection successful.")
except Exception as e:
    print(f"Error connecting to database: {e}")
    print("Please check MySQL server status, credentials, permissions, and required packages.")
    sys.exit(1)

# --- Training Vanna (Run deliberately when needed) ---
print("\n--- Vanna Training ---")
print("Loads metadata (DDL, docs, SQL examples) into Vanna's local ChromaDB vector store.")
train_vanna = True # Set to True to run training, False to skip for normal use
if train_vanna:
    df_information_schema = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")
    Plan = vn.get_training_plan_generic(df_information_schema)
    vn.train(plan=Plan)
    print("Running Vanna training...")

    # === Add your specific training calls here ===
    print("  Example: Fetching DDL (Customize for your setup)...")
    # df_ddl = vn.run_sql(f"SHOW CREATE TABLE your_table_name;")
    # if not df_ddl.empty: vn.train(ddl=df_ddl.iloc['Create Table'])

    print("  Example: Adding Documentation (Add yours)...")
    # --- Customer Table Documentation ---
    vn.train(documentation="Table CUSTOMER contains information about business customers.")
    vn.train(documentation="CUSTOMER.id is the unique identifier for each customer.")
    vn.train(documentation="CUSTOMER.name is the company name of the customer.")
    vn.train(documentation="CUSTOMER.addr_str contains the street address for the customer.")
    vn.train(documentation="CUSTOMER.addr_city contains the city for the customer.")
    vn.train(documentation="CUSTOMER.addr_ste contains the state for the customer.")
    vn.train(documentation="CUSTOMER.addr_ctry contains the country for the customer.")
    vn.train(documentation="CUSTOMER.addr_zip contains the zip-code for the customer.")

    # --- Vendor Table Documentation ---
    vn.train(documentation="Table VENDOR lists suppliers of parts.")
    vn.train(documentation="VENDOR.id is the unique identifier for each vendor.")
    vn.train(documentation="VENDOR.name is the name of the vendor/supplier.")
    vn.train(documentation="CUSTOMER.addr_str contains the street address for the vendor.")
    vn.train(documentation="CUSTOMER.addr_city contains the city for the vendor.")
    vn.train(documentation="CUSTOMER.addr_ste contains the state for the vendor.")
    vn.train(documentation="CUSTOMER.addr_ctry contains the country for the vendor.")
    vn.train(documentation="CUSTOMER.addr_zip contains the zip-code for the vendor.")

    # --- Part Master Table Documentation ---
    vn.train(documentation="Table PART_MST is the master list of all parts.")
    vn.train(documentation="PART_MST.id is the unique identifier which indicates the unique id for the part.")
    vn.train(documentation="PART_MST.p_code is the unique product code for the part.")
    vn.train(documentation="PART_MST.procurement indicates if a part is manufactured internally ('M') or purchased ('B'). It must be either 'M' or 'B'.")
    vn.train(documentation="PART_MST.unit_price represents the standard price per unit of the part.")
    vn.train(documentation="PART_MST.pref_order_qty specifies The preferred quantity to order or manufacture at once, depending on if the part is M or B.")
    vn.train(documentation="PART_MST.unit_meas the unit of measure for this part; usually 'EA' for 'eaches'.")
    vn.train(documentation="PART_MST.mfg_lt is the lead time to manufacture this part, if it is manufactured. This must an 'M' part and positive.")
    vn.train(documentation="PART_MST.curr_stock the current amount of this part in stock in inventory")

    # --- Add Documentation for BOM_MST, RTG_MST, JOB_OPER, JOB_MATL, RECV_MST, RECV_LINE ---
    # Example:
    vn.train(documentation="Table bill_of_m defines the Bill of Materials, showing parent-child part relationships.")
    vn.train(documentation="bill_of_m.m_part links to the parent part code in PART_MST. The manufactured part which consumes c_part.")
    vn.train(documentation="BOM_MST.c_part links to the child component part code in PART_MST. The component part consumed by m_part.")
    vn.train(documentation="BOM_MST.qty_req is The quantity of c_part required to make one unit of m_part.")

    # --- Job Master Table Documentation ---
    vn.train(documentation="Table job_mst represents the master data for all the jobs.")
    vn.train(documentation="job_mst.id is the unique identifier for a job.")
    vn.train(documentation="job_mst.job_stat signifies the current status of the job as follows: 'Q' (in queue), 'C' (completed), 'O' (opened), 'X' (cancelled), or 'H' (on hold).")
    vn.train(documentation="job_mst.part is the part that this job is manufacturing. This can be found in the PART_MST table.")
    vn.train(documentation="job_mst.qty is the quantity of the part that this job is manufacturing.")
    vn.train(documentation="job_mst.job_rls is the time at which this job began work.")
    vn.train(documentation="job_mst.job_cls is the time at which this job was completed.")

    # --- Nonconform master table documentation ---
    vn.train(documentation="Table NONCONFORM_MST defines the Nonconformance report master data.")
    vn.train(documentation="NONCONFORM_MST.id is a unique id for this nonconformance report.")
    vn.train(documentation="NONCONFORM_MST.job signifies if this nonconformance report was issued for a job, the job it was issued for.")
    vn.train(documentation="NONCONFORM_MST.po_no signifies if this nonconformance report was issued for a purchase order, the purchase order it was issued for.")
    vn.train(documentation="NONCONFORM_MST.qty signifies the quantity of parts on this job/purchase order which were flagged for nonconformance issues.")



    # --- Sales Order Header Table Documentation ---
    vn.train(documentation="Table SALES_MST represents the header information for a customer sales order.")
    vn.train(documentation="SALES_MST.order_no is the unique identifier for a sales order.")
    vn.train(documentation="SALES_MST.order_date is the date the sales order was placed.")
    vn.train(documentation="SALES_MST.cust links to the CUSTOMER table for the customer who placed the order.")

    # --- Sales Order Line Table Documentation ---
    vn.train(documentation="Table SALES_LINE contains individual line items for each sales order in SALES_MST.")
    vn.train(documentation="SALES_LINE.order_no links back to the SALES_MST table.")
    vn.train(documentation="SALES_LINE.line_no is the line number of the line item under the given order_no.")
    vn.train(documentation="SALES_LINE.order_stat is the current status of the line item: 'C' (completed), 'O' (opened), or 'X' (cancelled).")
    vn.train(documentation="SALES_LINE.part links to the PART_MST table for the specific part ordered.")
    vn.train(documentation="SALES_LINE.due is the requested delivery date for this line item.")
    vn.train(documentation="SALES_LINE.unit_price is the agreed price per unit for the part on this order line.")
    vn.train(documentation="SALES_LINE.qty is the number of units ordered for this part on this line item.")
    vn.train(documentation="SALES_LINE.line_cls is the date on which this line item was closed.")

    # --- Purchase Order Header Table Documentation ---
    vn.train(documentation="Table PURCHASE_MST represents the header information for a purchase order sent to a vendor.")
    vn.train(documentation="PURCHASE_MST.order_no is the unique identifier for a purchase order.")
    vn.train(documentation="PURCHASE_MST.order_date is the date the purchase order was created.")
    vn.train(documentation="PURCHASE_MST.vendor links to the VENDOR table for the supplier receiving the order.")

    # --- Purchase Order Line Table Documentation ---
    vn.train(documentation="Table PURCHASE_LINE contains individual line items for each purchase order in PURCHASE_MST.")
    vn.train(documentation="PURCHASE_LINE.order_no links back to the PURCHASE_MST table.")
    vn.train(documentation="PURCHASE_LINE.part links to the PART_MST table for the specific part being purchased.")
    vn.train(documentation="PURCHASE_LINE.due is the expected delivery date for this purchased part.")
    vn.train(documentation="PURCHASE_LINE.unit_cost is the cost per unit for the part on this purchase order line.")
    vn.train(documentation="PURCHASE_LINE.qty is the number of units purchased. It must be a positive number.")
    vn.train(documentation="PURCHASE_LINE.order_stat is the current status of the purchase order: 'C' (completed), 'O' (opened), or 'X' (cancelled).")
    vn.train(documentation="PURCHASE_LINE.order_type is the type of purchase order: either an 'S' for service order or 'M' for material order.")
    vn.train(documentation="PURCHASE_LINE.line_no is the line number of this purchase order line under the given order_no.")
    vn.train(documentation="PURCHASE_LINE.line_cls is the date on which this purchase order was closed.")


    # --- Vendor Part Table Documentation ---
    vn.train(documentation="Table VEND_PART represents vendor and part relationship.")
    vn.train(documentation="VEND_PART.vendor is the unique identifier for each vendor.")
    vn.train(documentation="VEND_PART.part is the part that can be procured from the vendor.")
    vn.train(documentation="VEND_PART.unit_cost is typically the creation or start date of the job.")
    vn.train(documentation="VEND_PART.part_lt is The standard lead time for which to place orders for part to vendor")


    print("Adding SQL examples...")
    vn.train(question="Get all customers", 
                sql="SELECT * FROM customer;")
    
    vn.train(question="Get all vendors in a specific state", 
                sql="SELECT *  FROM vendor WHERE 'addr_ste' = 'TX';")
    
    vn.train(question="Join sales orders with customer info", 
                sql="""SELECT s.order_no, s.order_date, c.name, c.addr_city FROM sales_mst s JOIN customer c ON s.cust = c.id;""")
    
    vn.train(question="Parts with their vendor and unit cost", 
                sql="""SELECT P.p_code, V.name AS vendor_name, VP.unit_cost FROM vend_part VP JOIN vendor  V ON VP.vendor = V.id JOIN part_mst P ON VP.part = P.id;""")
    
    vn.train(question="Total number of sales per customer", 
                sql="""SELECT C.name, COUNT(S.order_no) AS total_orders FROM customer C JOIN sales_mst S ON S.cust = C.id GROUP BY C.name order by COUNT(S.order_no) desc; """)
    
    vn.train(question="Average unit price of each part", 
                sql="""SELECT p_code, AVG(unit_price) AS avg_price FROM part_mst GROUP BY p_code order by AVG(unit_price);""")
    
    vn.train(question="Jobs released in last 30 days", 
                sql="""SELECT * FROM job_mst where date(job_rls)>= date_sub(curdate() , interval 30 day);""")
    
    vn.train(question="Purchase orders due next week", 
                sql="""SELECT * FROM purchase_line where week(due) = week(date_add(curdate() , interval 7 day));""")
    
    vn.train(question="All open jobs", 
                sql="""SELECT * FROM JOB_MST  WHERE job_stat = 'Q';""")
    
    vn.train(question="All sales lines with ‘COMPLETE’ status", 
                sql="""SELECT * FROM sales_line WHERE order_stat = 'C';""")
    
    vn.train(question="Get purchase line for specific order and line", 
                sql="""SELECT * FROM purchase_lineWHERE order_no = 'PO-5436934';""")
    
    vn.train(question="Customers with more than 3 orders", 
                sql="""SELECT name FROM CUSTOMER WHERE id IN (SELECT cust FROM SALES_MST GROUP BY cust HAVING COUNT(order_no) > 3);""")
    
    vn.train(question="List Sales Orders with Customer Names", 
                sql="""SELECT s.order_no, s.order_date, c.name AS customer_name, c.addr_city AS customer_city FROM SALES_MST s JOIN CUSTOMER c ON s.cust = c.id;""")
    
    vn.train(question="Show Details of Parts on a Specific Sales Order", 
                sql="""SELECT sl.order_no, sl.line_no,p.p_code AS part_code,p.unit_meas,sl.qty AS quantity_ordered,sl.unit_price AS price_on_order, p.unit_price AS current_part_price, sl.order_stat AS line_status FROM SALES_LINE sl JOIN PART_MST p ON sl.part = p.id where order_no = "SO-1490003"; """)
    
    vn.train(question="Calculate Total Quantity Sold for Each Part", 
                sql="""SELECT p.p_code, p.id AS part_id, SUM(sl.qty) AS total_quantity_sold FROM SALES_LINE sl JOIN PART_MST p ON sl.part = p.id WHERE sl.order_stat = 'C'  GROUP BY p.p_code, p.id ORDER BY total_quantity_sold DESC;""")
    
    vn.train(question="Calculate Total Value of Purchases per Vendor", 
                sql="""SELECT v.name AS vendor_name, COUNT(DISTINCT pl.order_no) AS number_of_pos, SUM(pl.qty * pl.unit_cost) AS total_purchase_value FROM PURCHASE_LINE pl JOIN PURCHASE_MST pm ON pl.order_no = pm.order_no JOIN VENDOR v ON pm.vendor = v.id GROUP BY v.name ORDER BY total_purchase_value DESC;""")
    
    vn.train(question="Find Customers and Vendors Located in the Same State", 
                sql="""SELECT c.name AS customer_name, v.name AS vendor_name, c.addr_ste AS state FROM CUSTOMER c JOIN VENDOR v ON c.addr_ste = v.addr_ste ORDER BY c.addr_ste, c.name, v.name;""")
    
    vn.train(question="Compare Current Stock vs. Total Purchased vs. Total Sold (Simplified)", 
                sql="""SELECT p.id, p.p_code, p.curr_stock AS current_stock_on_hand, COALESCE(pp.total_purchased, 0) AS total_units_purchased, COALESCE(ps.total_sold, 0) AS total_units_sold, (p.curr_stock + COALESCE(pp.total_purchased, 0) - COALESCE(ps.total_sold, 0)) AS calculated_stock_balance  FROM PART_MST p LEFT JOIN PartPurchases pp ON p.id = pp.part LEFT JOIN PartSales ps ON p.id = ps.part ORDER BY p.p_code; """)
    
    vn.train(question="List Parts Supplied by a Specific Vendor", 
                sql="""SELECT v.id, v.name AS vendor_name, p.p_code AS part_code, vp.unit_cost AS vendor_unit_cost, vp.part_lt AS vendor_lead_time FROM VEND_PART vp left JOIN VENDOR v ON vp.vendor = v.id left JOIN PART_MST p ON vp.part = p.id where v.id = 7 ORDER BY p.p_code;""")
    
    vn.train(question="Top 5 Customers by Sales Value in the Last 6 Months", 
                sql="""SELECT c.id AS customer_id, c.name AS customer_name, SUM(sl.qty * sl.unit_price) AS total_spent FROM CUSTOMER c JOIN SALES_MST s ON c.id = s.cust JOIN SALES_LINE sl ON s.order_no = sl.order_no WHERE s.order_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH) AND sl.order_stat <> 'X' GROUP BY c.id, c.name ORDER BY total_spent DESC LIMIT 5;""")
    
    vn.train(question="Parts Below Preferred Order Quantity", 
                sql="""SELECT id,p_code,curr_stock, pref_order_qty FROM PART_MST WHERE curr_stock < pref_order_qty AND procurement <> 'S' ORDER BY (pref_order_qty - curr_stock) DESC;""")
    
    vn.train(question="Monthly Sales Revenue Trend", 
                sql="""SELECT DATE_FORMAT(s.order_date, '%Y-%m') AS sales_month, round(SUM(sl.qty * sl.unit_price),2) AS monthly_revenue FROM SALES_MST s JOIN SALES_LINE sl ON s.order_no = sl.order_no WHERE sl.order_stat <> 'X' GROUP BY sales_month ORDER BY sales_month;""")

    vn.train(question="Direct Components for a Manufactured Part (Bill of Materials)", 
                sql="""SELECT mp.p_code AS manufactured_part_code, cp.p_code AS component_part_code, bom.qty_req AS quantity_required, cp.unit_meas AS component_unit FROM BILL_OF_M bom JOIN PART_MST mp ON bom.m_part = mp.id  JOIN PART_MST cp ON bom.c_part = cp.id  WHERE bom.m_part = 5638;""")
    
    vn.train(question="Total Component Quantity Required for Open Production Jobs", 
                    sql="""SELECT bom.c_part AS component_part_id, p.p_code AS component_part_code, SUM(j.qty * bom.qty_req) AS total_required_for_open_jobs FROM JOB_MST j JOIN BILL_OF_M bom ON j.part = bom.m_part JOIN PART_MST p ON bom.c_part = p.id   WHERE j.job_stat IN ('O', 'Q') GROUP BY bom.c_part, p.p_code ORDER BY total_required_for_open_jobs DESC;""")
        
    vn.train(question="Nonconformance Rate per Part (Simplified: NC Qty / Qty Purchased)", 
                    sql="""WITH PartPurchasesReceived AS (SELECT pl.part, SUM(pl.qty) AS total_received FROM PURCHASE_LINE pl WHERE pl.order_stat = 'C' GROUP BY pl.part),PartNonconformance AS (SELECT pl.part, SUM(nc.qty) AS total_nonconforming FROM NONCONFORM_MST nc JOIN PURCHASE_MST pm ON nc.po_no = pm.order_no  JOIN PURCHASE_LINE pl ON pm.order_no = pl.order_no AND nc.job IS NULL  GROUP BY pl.part)SELECT p.id AS part_id, p.p_code, COALESCE(ppr.total_received, 0) AS total_received, COALESCE(pnc.total_nonconforming, 0) AS total_nonconforming, CASE WHEN COALESCE(ppr.total_received, 0) > 0 THEN (COALESCE(pnc.total_nonconforming, 0) / ppr.total_received) * 100 ELSE 0 END AS nonconformance_rate_percent FROM PART_MST p LEFT JOIN PartPurchasesReceived ppr ON p.id = ppr.part LEFT JOIN PartNonconformance pnc ON p.id = pnc.part WHERE p.procurement = 'B'  ORDER BY nonconformance_rate_percent DESC;""")
    
    vn.train(question="Customers Who Haven't Ordered in Over a Year", 
                    sql="""SELECT c.id, c.name, MAX(s.order_date) AS last_order_date  FROM CUSTOMER c LEFT JOIN SALES_MST s ON c.id = s.cust AND s.order_date >= DATE_SUB(CURDATE(), INTERVAL 1 YEAR) WHERE s.order_no IS NULL  GROUP BY c.id, c.name  ORDER BY c.name;""")
    
    vn.train(question="Parts Purchased from Multiple Vendors with Cost Comparison", 
                    sql="""SELECT p.id AS part_id, p.p_code, COUNT(DISTINCT vp.vendor) AS number_of_vendors, MIN(vp.unit_cost) AS min_vendor_cost, MAX(vp.unit_cost) AS max_vendor_cost, (MAX(vp.unit_cost) - MIN(vp.unit_cost)) AS cost_difference FROM PART_MST p JOIN VEND_PART vp ON p.id = vp.part GROUP BY p.id, p.p_code HAVING COUNT(DISTINCT vp.vendor) > 1  ORDER BY cost_difference DESC, p.p_code;""")
    
    

    # --- General Business Rules/Concepts ---
    # vn.train(documentation="'Lead Time' generally refers to the time required to procure or produce a part (PART_MST.sys_lt is a default).")
    # vn.train(documentation="'On-time Delivery' can be assessed by comparing SALES_LINE.due_date with actual delivery dates (potentially in another table like deliveries if it exists).")



# def extract_sql_from_response(response: str) -> str:
#     """Extract SQL code from LLM response using regex patterns"""
#     # Match multi-line SQL between `````` with flexible whitespace
#     response.split(":\n\n")[1].replace("\n"," ")
# def create_plot(question,extracted_sql,df_result):
#     plotly_code = vn.generate_plotly_code(question = question,
#                                           sql = extracted_sql,
#                                           df = df_result)
#     fig = vn.get_plotly_figure(plotly_code=plotly_code,
#                                df=df_result)
#     fig.show()
def extract_sql_from_response(response: str) -> str:
    """Extract SQL code from LLM response using multi-pattern approach"""
    response = response.replace("\n", " ")
    # response = response.replace("sql", " ")
    # Pattern 1: Extract SQL from code blocks
    code_block_match = re.search(
        r'``````',
        response,
        re.DOTALL | re.IGNORECASE
    )
    if code_block_match:
        return code_block_match.group(1).strip()

    # Pattern 2: Extract SQL after colon + newlines
    colon_split = response.split(":\n\n")
    if len(colon_split) > 1:
        potential_sql = colon_split[-1].split(';')[0] + ';'
        # Verify it contains SQL keywords
        if re.search(r'\b(SELECT|UPDATE|INSERT|DELETE|WITH)\b', potential_sql, re.IGNORECASE):
            return potential_sql.replace('`', '').strip()

    # Pattern 3: Fallback to generic SQL detection
    generic_match = re.search(
        r'\b(SELECT|UPDATE|INSERT|DELETE|WITH)\b.*?;',
        response,
        re.DOTALL | re.IGNORECASE
    )
    return generic_match.group(0).strip() if generic_match else None

current_date = datetime.now().strftime("%Y-%m-%d")
# --- Main Interaction Loop ---
# --- Modified Main Interaction Loop ---
while True:
    try:
        question = (input("\nUser: "))
        
        if question.lower() == 'exit':
            break
                  
                    
        while True: 
            new_question = (f"""Today is {current_date}. You are a manufacturing analyst. "
                    Generate ONLY the SQL query (no comments), "
                    "for the following new_question:""" + f"{question}")
            print("\nAgent thinking...")
            start_time = time.time()
            
            # Get classified response
            llm_response = vn.generate_contextual_response(new_question)
            category = vn.classify_question(new_question)
            
            # Extract SQL using implemented function
            extracted_sql = extract_sql_from_response(llm_response)
            try:
                extracted_sql = extracted_sql.split("```sql")[1].replace("```","")
            except:
                continue    
            if not extracted_sql:
                print("\nCould not extract executable SQL from response")
                continue
                
            print(f"\n--- Extracted SQL ({category.upper()}) ---")
            print(extracted_sql)
            print("--- End of Extracted SQL ---")
            
            # Unified execution using Vanna's method
            execute = input(f"\nExecute {category.upper()} SQL query? (y/n): ").lower().strip()
            
            if execute == 'y':
                try:
                    print("Executing query...")
                    exec_start_time = time.time()
                    
                    df_result = vn.run_sql(sql=extracted_sql)
                    # Category-specific post-processing
                    if category == "judgement":
                        print("\n** Analytical Insights **")
                        print(df_result.describe())
                    elif category == "suggestion":
                        print("\n** Recommended Actions **")
                        print(df_result.head())
                    else:  # descriptive
                        print("\n** Query Results **")
                        
                    
                    # Display results
                    with pd.option_context('display.max_rows', 20, 
                                        'display.max_columns', None,
                                        'display.width', 1000):
                        if len(df_result) > 20:
                            print(df_result.head(20))
                            print(f"... (truncated, {len(df_result)} rows total)")
                        else:
                            print(df_result)


                    # create_plot(df_result,extracted_sql,question)
                    print(f"Query executed successfully in {time.time() - exec_start_time:.2f} seconds")
                    re_run = input(f"\Should I re-think {question.upper()} for a better response? (y/n): ").lower().strip()
                    if re_run == 'n':
                        save = input(f"\nDo You want to save this {category.upper()} SQL query for future use? (y/n): ").lower().strip()
                        if save == "y":
                            print("Saving results...")
                            vn.train(question=question, sql=extracted_sql)
                            break
                        else:
                            print("SQL query not saved.")
                            break    
                    else:
                        print("Retrying...")
                        continue
                        

                except Exception as exec_e:
                    print(f"\nError executing SQL: {type(exec_e).__name__} - {exec_e}")
                    print(f"Failed SQL:\n{extracted_sql}")
                    re_run = input(f"\nShould I re-think {question.upper()} for a better response? (y/n): ").lower().strip()
                    
                    if re_run == 'n':
                        break
                    else:
                        print("Retrying...")
                        continue

            else:
                break

    except KeyboardInterrupt:
        print("\nExiting...")
        break

###### TESTING CLASSIFICATION FUNCTION ######
   # Add this at the bottom of your script, after the training section
# def test_classify_question():
#     # Create an instance of MyLocalLlm
#     llm = MyLocalLlm()
    
#     # Test cases with expected classifications
#     test_cases = [
#         ("What is my projected revenue over the next month?", "descriptive"),
#         ("Why was order PO-123 late?", "judgement"),
#         ("How can I improve on-time delivery for customer X?", "suggestion"),
#         ("List all parts in inventory", "descriptive"),
#         ("Should I modify my system lead times?", "judgement"),
#         ("What should I do about late deliveries?", "suggestion")
#     ]
    
#     print("\n=== Testing classify_question function ===")
#     for question, expected in test_cases:
#         print(f"\nQuestion: {question}")
#         print(f"Expected classification: {expected}")
#         try:
#             result = llm.classify_question(question)
#             print(f"Actual classification: {result}")
#             print(f"Match: {'✓' if result == expected else '✗'}")
#         except Exception as e:
#             print(f"Error during classification: {str(e)}")
    
#     print("\n=== End of testing ===")

# # Run the test function
# if __name__ == "__main__":
#     test_classify_question()

    # top 10 customers this year with respect to sales value
    # , no explanations, no markdown, no text, no code fences