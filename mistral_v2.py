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
# try:
# import vanna
from vanna.base import VannaBase
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
import openai
import mysql.connector
import pandas as pd
# except ImportError as e:
# print(f"Import Error: {e}")
# sys.exit(1)

# --- Configuration ---
db_type = "mysql"
mysql_host = "127.0.0.1"
mysql_port = 3306
mysql_user = "root"
mysql_password = "Raekwon_wtc$36" 
mysql_dbname = "ft_database"
llm_model_name = 'mistral-7b-instruct-v0.1.Q8_0.gguf'
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
QUESTION_TYPES = ["descriptive", "judgement", "suggestion"]  
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
                max_tokens=kwargs.get('max_tokens', 5000),
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

    def get_similar_questions(self, question: str, n_results: int = 3) -> List[str]:
        """
        Search for similar questions in the training data.
        
        Args:
            question (str): The question to find similar matches for
            n_results (int): Number of similar questions to return
            
        Returns:
            List[str]: List of similar questions
        """
        try:
            # Get all training data
            training_data = self.get_training_data()
            if not training_data:
                return []

            # Convert question to lowercase for case-insensitive comparison
            question_lower = question.lower()
            
            # Find similar questions using simple text matching
            similar_questions = []
            for item in training_data:
                if 'question' in item:
                    # Check if the question contains any key terms from the training data
                    if any(term in question_lower for term in item['question'].lower().split()):
                        similar_questions.append(item['question'])
                        if len(similar_questions) >= n_results:
                            break
            
            return similar_questions
        except Exception as e:
            print(f"Error in get_similar_questions: {str(e)}")
            return []

    def get_sql_for_question(self, question: str) -> str:
        """
        Get the SQL query associated with a specific question from training data.
        
        Args:
            question (str): The question to get SQL for
            
        Returns:
            str: The SQL query if found, None otherwise
        """
        try:
            # Get all training data
            training_data = self.get_training_data()
            if not training_data:
                return None

            # Convert question to lowercase for case-insensitive comparison
            question_lower = question.lower()
            
            # Find the most similar question and its SQL
            best_match = None
            best_match_score = 0
            
            for item in training_data:
                if 'question' in item and 'sql' in item:
                    # Calculate similarity score based on common words
                    training_question = item['question'].lower()
                    common_words = set(question_lower.split()) & set(training_question.split())
                    score = len(common_words)
                    
                    if score > best_match_score:
                        best_match_score = score
                        best_match = item['sql']
            
            return best_match if best_match_score > 0 else None
        except Exception as e:
            print(f"Error in get_sql_for_question: {str(e)}")
            return None

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
    df_information_schema = vn.run_sql(f"SELECT * FROM INFORMATION_SCHEMA.COLUMNS where TABLE_SCHEMA = '{mysql_dbname}';")
    Plan = vn.get_training_plan_generic(df_information_schema)
    vn.train(plan=Plan)
    table_list = df_information_schema['TABLE_NAME'].unique().tolist()
    column_list = df_information_schema['COLUMN_NAME'].unique().tolist()
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


    vn.train(documentation="""Comprehensive Chapter-by-Chapter Report on Factory Physics by Wallace J. Hopp and Mark L. Spearman

Factory Physics provides a rigorous framework for understanding manufacturing and supply chain operations through scientific principles. Below is a detailed summary of each chapter, structured to reflect the book's three-part organization: historical context, foundational theories, and practical applications. Key formulas and laws are described in narrative form to align with the requested format.
Part I: The Lessons of History
Chapter 1: Manufacturing in America
This chapter traces the evolution of U.S. manufacturing from the Industrial Revolution to modern practices. It critiques the decline of American manufacturing dominance in the late 20th century, attributing it to short-term financial strategies, overreliance on technology, and fragmented management approaches. The authors emphasize the need for a scientific foundation to restore competitiveness, setting the stage for the "Factory Physics" framework.
Chapter 2: Inventory Control: From EOQ to ROP
The chapter reviews classical inventory models, including the Economic Order Quantity (EOQ) model, which balances ordering and holding costs to determine optimal batch sizes. The Reorder Point (ROP) method is introduced as a way to manage stochastic demand by setting inventory thresholds based on lead times and service levels. Criticisms of these models include their reliance on static assumptions and neglect of variability.
Chapter 3: The MRP Crusade
Material Requirements Planning (MRP) and its evolution into Enterprise Resource Planning (ERP) are analyzed. While MRP improved coordination of production schedules, its limitations-such as infinite capacity assumptions and sensitivity to demand fluctuations-are highlighted. The authors argue that MRP's rigidity often leads to inefficiencies in dynamic environments.
Chapter 4: From the JIT Revolution to Lean Manufacturing
The Just-in-Time (JIT) philosophy and its successor, Lean Manufacturing, are explored. Key principles like waste reduction, pull systems, and Total Quality Management (TQM) are discussed. However, the chapter warns against dogmatic adherence to Lean without understanding underlying variability. The Kanban system is presented as a method to limit work-in-process (WIP) but is noted to falter under high variability.
Chapter 5: What Went Wrong
This chapter diagnoses the failures of 20th-century manufacturing strategies, including overemphasis on financial metrics, disjointed improvement initiatives, and inadequate integration of human factors. The authors advocate for a systems-oriented approach grounded in the laws of factory physics.
Part II: Factory Physics – The Science of Manufacturing
Chapter 6: A Science of Manufacturing
The authors introduce the core philosophy of Factory Physics: manufacturing systems obey natural laws akin to physical systems. Key objectives-throughput, inventory, and cycle time-are defined, and the need for predictive models to balance these metrics is emphasized.
Chapter 7: Basic Factory Dynamics
Foundational relationships are established:
Little's Law: Work-in-process (WIP) equals the product of throughput and cycle time.
Bottleneck Rate: The slowest process in a system determines maximum throughput.
Critical WIP: The minimum WIP required to achieve bottleneck throughput.
Chapter 8: Variability Basics
Variability is quantified using the coefficient of variation (CV). The chapter explains how randomness in processing times and demand propagates through systems, degrading performance. The Kingman equation (VUT) is introduced, linking cycle time to variability, utilization, and process time.
Chapter 9: The Corrupting Influence of Variability
The Variability Law states that increased variability always degrades system performance, necessitating buffers in inventory, capacity, or time. The Worst-Case Performance Law shows that deterministic systems can exhibit poor performance due to misaligned scheduling, challenging Lean's aversion to all variability.
Chapter 10: Push and Pull Production Systems
Push systems (e.g., MRP) and pull systems (e.g., CONWIP, Kanban) are compared. The CONWIP method, which limits WIP while maintaining flexibility, is advocated as a hybrid approach. The authors demonstrate how pull systems reduce cycle time variability by controlling WIP.
Chapter 11: The Human Element
Human factors-such as motivation, decision-making biases, and the balance of responsibility-are analyzed. The chapter argues that effective systems design must align incentives with operational goals to avoid counterproductive behaviors.
Chapter 12: Total Quality Manufacturing
Quality management principles, including Statistical Process Control (SPC) and Six Sigma, are integrated into the Factory Physics framework. The cost of quality is framed as a trade-off between prevention costs and failure costs.
Part III: Principles in Practice
Chapter 13: A Pull Planning Framework
A hierarchical planning approach is proposed, combining long-term forecasts with short-term pull signals. The quota-setting model aligns production targets with capacity constraints to avoid overburdening bottlenecks.
Chapter 14: Shop Floor Control
Practical strategies for managing shop floors include CONWIP configurations and Statistical Throughput Control (STC), which monitors output to detect deviations from planned performance.
Chapter 15: Production Scheduling
Scheduling techniques prioritize bottleneck resources using dynamic dispatching rules. The earliest due date (EDD) and critical ratio (CR) methods are evaluated for their ability to balance throughput and timely delivery.
Chapter 16: Aggregate and Workforce Planning
The chapter integrates capacity planning with workforce management. Linear programming models optimize product mix and labor allocation under demand uncertainty.
Chapter 17: Supply Chain Management
Inventory management strategies for raw materials, WIP, and finished goods are detailed. The multi-echelon supply chain model emphasizes coordination across stages to minimize bullwhip effects.
Chapter 18: Capacity Management
Methods for capacity expansion and line balancing are discussed. The line-of-balance problem is solved to synchronize production rates across stations, ensuring smooth flow.
Chapter 19: Synthesis – Pulling It All Together
The final chapter synthesizes concepts into a unified strategy. Case studies illustrate how combining Lean, Six Sigma, and Factory Physics principles achieves sustainable improvements. The authors stress the importance of experimentation and data-driven decision-making.
Key Formulas in Narrative Form
Little's Law: The average number of items in a system equals the average arrival rate multiplied by the average time each item spends in the system.
Kingman's Equation (VUT): Cycle time increases with variability, utilization, and process time.
Bottleneck Rate: The maximum throughput of a system is determined by its slowest process.
Variability Buffering: Systems buffer variability using inventory, capacity, or time.
CONWIP WIP Limit: The optimal WIP level balances throughput and cycle time.
By grounding manufacturing management in scientific principles, Factory Physics provides a timeless framework for optimizing complex operations. Its integration of theory and practice makes it indispensable for academics and industry professionals alike.""")


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
    
    vn.train(question="All sales lines with 'COMPLETE' status", 
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
    
    vn.train(question="Implement simple exponential smoothing forecast",
            sql = "WITH MonthlySales AS (SELECT  DATE_FORMAT(order_date, '%Y-%m') AS month, SUM(qty) AS sales FROM SALES_LINE join sales_mst on SALES_LINE.order_no = sales_mst.order_no GROUP BY DATE_FORMAT(order_date, '%Y-%m')),Smoothing AS (SELECT month, sales, @forecast := COALESCE(@forecast * 0.8 + sales * 0.2, sales) AS forecast FROM MonthlySales CROSS JOIN (SELECT @forecast := NULL) init ORDER BY month)SELECT * FROM Smoothing;" )
    
    vn.train(question="Compare actual purchase costs vs vendor quoted prices.",
            sql="SELECT pl.order_no,p.p_code,pl.unit_cost AS actual_cost,vp.unit_cost AS quoted_cost,(pl.unit_cost - vp.unit_cost) AS variance FROM PURCHASE_LINE pl JOIN VEND_PART vp ON pl.part = vp.part AND pl.order_no IN (SELECT order_no FROM PURCHASE_MST WHERE vendor in (select vendor from vend_part))JOIN PART_MST p ON pl.part = p.id;")
    
    vn.train(question="Analyze sales distribution by customer state.",
            sql="SELECT c.addr_ste AS state,COUNT(DISTINCT s.order_no) AS order_count,SUM(sl.qty * sl.unit_price) AS total_revenue FROM CUSTOMER c JOIN SALES_MST s ON c.id = s.cust JOIN SALES_LINE sl ON s.order_no = sl.order_no GROUP BY c.addr_ste ORDER BY total_revenue DESC;")
    
    vn.train(question="Calculate yield loss percentage from non-conformances.",
                sql="SELECT p.p_code, COALESCE(SUM(n.qty),0) AS nc_qty, SUM(j.qty) AS total_produced, (COALESCE(SUM(n.qty),0) / SUM(j.qty)) * 100 AS yield_loss_percent FROM JOB_MST j LEFT JOIN NONCONFORM_MST n ON j.id = n.job JOIN PART_MST p ON j.part = p.id GROUP BY p.id,p.p_code HAVING yield_loss_percent > 5;")
    
    vn.train(question="Identify single-source vendors with high-risk in terms of part quantity.",
                sql="select name,p_code,curr_stock,pref_order_qty from (SELECT  p.id, p.p_code, COUNT(DISTINCT vp.vendor) AS vendor_count, sum(p.curr_stock) curr_stock, sum(p.pref_order_qty) pref_order_qty FROM VEND_PART vp JOIN PART_MST p ON vp.part = p.id JOIN VENDOR v ON vp.vendor = v.id GROUP BY p.id,p.p_code HAVING vendor_count = 1 ) t join vend_part on t.id = vend_part.part join vendor on vend_part.vendor = vendor.id where curr_stock < pref_order_qty;")
    
    vn.train(question="Categorize inventory by last movement date.",
                    sql="SELECT p.p_code, sum(p.curr_stock) curr_stock, CASE WHEN MAX(s.order_date) IS NULL THEN 'No Sales' WHEN MAX(s.order_date) < DATE_SUB(NOW(), INTERVAL 6 MONTH) THEN 'Slow Moving' ELSE 'Active' END AS inventory_status FROM PART_MST p LEFT JOIN SALES_LINE sl ON p.id = sl.part LEFT JOIN SALES_MST s ON sl.order_no = s.order_no GROUP BY p.p_code;")
    
    vn.train(question="Find frequently paired parts in customer orders.",
                    sql="SELECT a.part AS part1, b.part AS part2, COUNT(*) AS pair_count FROM SALES_LINE a JOIN SALES_LINE b ON a.order_no = b.order_no AND a.part < b.part GROUP BY part1, part2 HAVING pair_count > 1 ORDER BY pair_count DESC;")
    
    vn.train(question="Calculate average delay between job release and completion.",
                    sql="SELECT p.p_code,AVG(DATEDIFF(j.job_cls, j.job_rls)) AS avg_delay_days, COUNT(*) AS job_count FROM JOB_MST j JOIN PART_MST p ON j.part = p.id WHERE j.job_stat = 'C' GROUP BY p.p_code HAVING avg_delay_days > 7;")
    
    vn.train(question="Identify customers with decreasing order frequency over quarters.",
                    sql="WITH QuarterlyOrders AS (SELECT c.id, QUARTER(s.order_date) AS qtr, COUNT(*) AS orders FROM CUSTOMER c JOIN SALES_MST s ON c.id = s.cust GROUP BY c.id, QUARTER(s.order_date))SELECT id AS customer_id,qtr,orders, LAG(orders) OVER(PARTITION BY id ORDER BY qtr) AS prev_qtr_orders,(orders - LAG(orders) OVER(PARTITION BY id ORDER BY qtr)) AS trend FROM QuarterlyOrders;")
    
    vn.train(question="Find parts with no sales in the last 6 months but maintained inventory.",
                    sql="SELECT p.p_code, p.curr_stock, MAX(s.order_date) AS last_sale_date FROM PART_MST p LEFT JOIN SALES_LINE sl ON p.id = sl.part LEFT JOIN SALES_MST s ON sl.order_no = s.order_no GROUP BY p.p_code,p.curr_stock HAVING last_sale_date < DATE_SUB(NOW(), INTERVAL 6 MONTH) OR last_sale_date IS NULL;")
    
    vn.train(question="Predict next month's sales using 3-month moving average.",
                    sql="WITH MonthlySales AS (SELECT DATE_FORMAT(s.order_date, '%Y-%m') AS month, SUM(sl.qty) AS total_sales FROM SALES_MST s JOIN SALES_LINE sl ON s.order_no = sl.order_no GROUP BY DATE_FORMAT(s.order_date, '%Y-%m'))SELECT month, total_sales, AVG(total_sales) OVER(ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS forecast FROM MonthlySales;")
    
    vn.train(question="Identify parts with inconsistent vendor lead times exceeding 20% variability.",
                    sql="SELECT p.p_code,v.id, v.name AS vendor,AVG(CAST(SUBSTRING_INDEX(vp.part_lt, ' ', 1) AS UNSIGNED)) AS avg_lead_days,STDDEV(CAST(SUBSTRING_INDEX(vp.part_lt, ' ', 1) AS UNSIGNED)) AS std_deviation, (STDDEV(CAST(SUBSTRING_INDEX(vp.part_lt, ' ', 1) AS UNSIGNED)) /  AVG(CAST(SUBSTRING_INDEX(vp.part_lt, ' ', 1) AS UNSIGNED))) AS variability_ratio FROM VEND_PART vp JOIN VENDOR v ON vp.vendor = v.id JOIN PART_MST p ON vp.part = p.id GROUP BY v.id,p.p_code, v.name;")
    
    vn.train(question="Suggest price adjustments based on vendor cost changes and sales performance.",
                    sql="SELECT p.id, p.p_code, p.unit_price AS current_price,MIN(vp.unit_cost) AS lowest_vendor_cost,AVG(sl.unit_price) AS avg_sale_price, CASE WHEN MIN(vp.unit_cost) > p.unit_price * 0.9 THEN 'Increase Price' WHEN AVG(sl.unit_price) < p.unit_price * 0.8 THEN 'Decrease Price' ELSE 'Maintain Price' END AS recommendation FROM PART_MST p JOIN VEND_PART vp ON p.id = vp.part JOIN SALES_LINE sl ON p.id = sl.part GROUP BY p.id,p.p_code, p.unit_price;")
    
    vn.train(question="Find components with the highest demand from open production jobs compared to current stock.",
            sql="WITH ComponentDemand AS (SELECT bom.c_part, SUM(j.qty * bom.qty_req) AS required_qty FROM JOB_MST j JOIN BILL_OF_M bom ON j.part = bom.m_part WHERE j.job_stat IN ('Q','O') GROUP BY bom.c_part)SELECT p.p_code,cd.required_qty,p.curr_stock,(cd.required_qty - p.curr_stock) AS deficit FROM ComponentDemand cd JOIN PART_MST p ON cd.c_part = p.id WHERE cd.required_qty > p.curr_stock;")
    
    vn.train(question="Calculate the financial impact of non-conformances by linking them to purchase order costs.",
            sql = "SELECT n.id AS nc_id,pl.order_no,p.p_code,n.qty AS nc_quantity,pl.unit_cost,(n.qty * pl.unit_cost) AS financial_impact FROM NONCONFORM_MST n JOIN PURCHASE_MST pm ON n.po_no = pm.order_no JOIN PURCHASE_LINE pl ON pm.order_no = pl.order_no JOIN PART_MST p ON pl.part = p.idWHERE n.po_no IS NOT NULL;")
    
    vn.train(question="Rank vendors by average delivery speed (time between order date and line closure) for completed purchase orders.",
            sql="SELECT v.name,AVG(DATEDIFF(pl.line_cls, pm.order_date)) AS avg_delivery_days,DENSE_RANK() OVER(ORDER BY AVG(DATEDIFF(pl.line_cls, pm.order_date))) AS performance_rank FROM PURCHASE_MST pm JOIN PURCHASE_LINE pl ON pm.order_no = pl.order_no JOIN VENDOR v ON pm.vendor = v.id WHERE pl.order_stat = 'C' GROUP BY v.id,v.name HAVING avg_delivery_days IS NOT NULL;")
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

# --- SQL Extraction Function ---
def extract_sql_from_response(response: str) -> str:
    """Extract SQL code from LLM response using multi-pattern approach"""
    if not response:
        return None
        
    # Clean up the response
    response = response.strip()
    
    # Pattern 1: Look for SQL between ```sql and ``` markers
    sql_block_pattern = r'```sql\s*(.*?)\s*```'
    sql_matches = re.findall(sql_block_pattern, response, re.DOTALL)
    if sql_matches:
        return sql_matches[0].strip()
    
    # Pattern 2: Look for SQL between ``` markers
    code_block_pattern = r'```\s*(.*?)\s*```'
    code_matches = re.findall(code_block_pattern, response, re.DOTALL)
    if code_matches:
        for match in code_matches:
            if re.search(r'\b(SELECT|INSERT|UPDATE|DELETE|WITH)\b', match, re.IGNORECASE):
                return match.strip()
    
    # Pattern 3: Direct SQL pattern
    sql_pattern = r'\b(SELECT|WITH|INSERT|UPDATE|DELETE)[\s\S]+?;'
    direct_match = re.search(sql_pattern, response, re.IGNORECASE)
    if direct_match:
        return direct_match.group(0).strip()
    
    return None

# --- complex query architecture --- 
def generate_complex_sql(question: str) -> str:
    """Handles multi-step manufacturing queries"""
    system_msg = """You are an SQL expert. Follow these steps:
    1. Identify required tables from: sales_line, part_mst, purchase_line
    2. Determine necessary joins and filters
    3. Apply temporal constraints using CURDATE()
    4. Include proper aggregations (SUM, AVG, COUNT)
    5. Validate against database schema"""
    
    return vn.generate_sql(
        question=question,
        chat_history=[vn.system_message(system_msg)]
    )


# --- Question Refinement Function ---
def refine_question(original_question: str) -> str:
    refinement_prompt = f"""You are an expert Prompt Engineer. Rephrase and expand this manufacturing question to be more specific and actionable. 
    Maintain original intent while adding context about tables from {table_list} and columns from {column_list}. Return ONLY the improved question.

    Original: {original_question}
    Refined: """
    
    refined = vn.submit_prompt(refinement_prompt).strip()
    return refined if len(refined) > len(original_question) else original_question
    

# --- LLM Analysis and Explanation ---
def analyze_and_explain(question, df_result, category):
    """
    For judgement/suggestion: Use LLM to analyze the DataFrame and give a judgement or recommendation.
    For descriptive: Optionally summarize the data.
    """
    # Convert DataFrame to CSV or dict for LLM context (limit rows for brevity)
    data_sample = df_result.head(20).to_csv(index=False)
    if category == "judgement":
        prompt = f"""
You are a manufacturing analytics expert.
Given the following question:
"{question}"
And the following data (CSV format):
{data_sample}

Analyze the data, provide a clear judgement about the situation, and explain your reasoning. Clearly state: "Judgement: ...". Advise the user to independently verify your judgement.
"""
    elif category == "suggestion":
        prompt = f"""
You are a manufacturing analytics expert.
Given the following question:
"{question}"
And the following data (CSV format):
{data_sample}

Analyze the data and provide a concrete, actionable recommendation. Clearly state: "Suggestion: ...". Advise the user to independently verify your suggestion.
"""
    else:
        # For descriptive, just summarize (optional)
        prompt = f"""
You are a manufacturing analytics expert.
Given the following question:
"{question}"
And the following data (CSV format):
{data_sample}

Briefly summarize the main findings from the data.
"""
    # Call your LLM (reuse submit_prompt)
    response = vn.submit_prompt(prompt)
    return response.strip()


current_date = datetime.now().strftime("%Y-%m-%d")
# --- Main Interaction Loop ---
# --- Modified Main Interaction Loop ---

while True:
    try:
        question = (input("\nUser: "))
        
        if question.lower() == 'exit':
            break
        else:
            # new_question = refine_question(question)
            new_question = f"Remember Today's date for date related queries {current_date}. Answer this question: " + f" {question}"
        # Get classified response
        category = vn.classify_question(new_question)
        print(f"\n--- Classified Category: {category.upper()} ---")            
        
        
        while True:
            print("\nAgent thinking...")
            print(new_question)          
            start_time = time.time()

            llm_response = generate_complex_sql(new_question)

            # Extract SQL using implemented function
            try:
                extracted_sql = extract_sql_from_response(llm_response)
                try:
                    extracted_sql = extracted_sql.split("```sql")[1].replace("```","")
                    print(f"\n--- Extracted SQL ({category.upper()}) ---")
                    print(extracted_sql)
                    print("--- End of Extracted SQL ---")
                except:
                    extracted_sql = extract_sql_from_response(llm_response)    
            except:
                print("\nCould not extract executable SQL from response")
                continue

            if category!="suggestion":
                execute = input(f"\nExecute {category.upper()} SQL query? (y/n): ").lower().strip()
                
                if execute == 'y':
                    try:
                        print("Executing query...")
                        exec_start_time = time.time()
                        
                        df_result = vn.run_sql(sql=extracted_sql)
                        # Category-specific post-processing
                        if category == "judgement":
                            print("\n** Analytical Insights **")
                            # print(df_result.describe())
                            # print(f"\n=== {category.upper()} EXPLANATION ===\n{explanation}")

                        # elif category == "suggestion":
                        #     print("\n** Recommended Actions **")
                        #     print(df_result.head())
                            # explanation = analyze_and_explain(question, df_result, category)
                            

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
                        explanation = analyze_and_explain(question, df_result, category)
                        print(f"\n=== {category.upper()} EXPLANATION ===\n{explanation}")
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
                            new_question = (f"""The query: {extracted_sql} is giving this error:  {type(exec_e).__name__} - {exec_e} 
                                            resolve the error and provide a better response for this question: """
                                            + new_question)
                            continue
                            

                    except Exception as exec_e:
                        print(f"\nError executing SQL: {type(exec_e).__name__} - {exec_e}")
                        print(f"Failed SQL:\n{extracted_sql}")
                        re_run = input(f"\nShould I re-think {question.upper()} for a better response? (y/n): ").lower().strip()
                        
                        if re_run == 'n':
                            break
                        else:
                            print("Retrying...")
                            new_question = (f"""The query: {extracted_sql} is giving this error:  {type(exec_e).__name__} - {exec_e} 
                                            resolve the error and provide a better response for this question: """
                                            + question)
                            continue

                else:
                    break
            else:
                s_rethink = input(f"\nAre you satisfied with the SUGGESTION (y/n): ").lower().strip()
                if s_rethink == 'n':
                    break
                else:
                    print("Retrying...")
                    continue



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
    #  SHOULD WE CHANGE OUR VENDORS BASED ON THEIR LATE DELIVERIES?
    # whats the total sales value till now?
    # analyse the required data and recommend ways to optimize the supply chain



# # # --- Model Fine-Tuning and Deployment ---
# class ManufacturingFineTuner:  
#     def update_model(self, new_data: Dataset):  
#         # Apply LoRA on production data  
#         lora_config = LoraConfig(r=8, target_modules=["q_proj", "v_proj"])  
#         trainer = SFTTrainer(  
#             model=base_model,  
#             train_dataset=new_data,  
#             peft_config=lora_config  
#         )  
#         trainer.train()  
#         # Merge and quantize updated model  
#         self.model = merge_and_quantize(trainer.model)  