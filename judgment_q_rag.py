import time
import os
import sys
import warnings
import json
import re
import traceback # For detailed error printing
from typing import List, Dict, Any, TypedDict, Annotated, Sequence
import operator

# --- Environment Setup ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Package Imports and Checks ---
try:
    import vanna
    from vanna.base import VannaBase
    from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
    import openai
    import mysql.connector
    import pandas as pd
    # --- LangChain / LangGraph Imports ---
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage, SystemMessage
    from langchain_core.tools import tool
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode # Use the prebuilt ToolNode

    print("Required packages imported successfully.")
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure 'vanna[chromadb]', 'openai', 'mysql-connector-python', 'cryptography', 'pandas', 'llama-cpp-python', 'langchain', 'langchain_openai', 'langgraph', 'langchain_community' are installed.")
    sys.exit(1)

# --- Configuration ---
Xfrozen_modules= "off"
db_type = "mysql"
mysql_host = "127.0.0.1"
mysql_port = 3306
mysql_user = "root" # !!! REPLACE !!!
mysql_password = "Raekwon_wtc$36" # !!! REPLACE !!!
mysql_dbname = "ft_database" # !!! REPLACE !!!

llm_model_name = 'mistral-7b-instruct-v0.1.Q8_0.gguf'
local_server_url = 'http://127.0.0.1:8000/v1'
placeholder_api_key = "sk-no-key-required"

# --- Initialize Vanna Instance ---
vn = None
try:
    # Define Vanna's LLM Interface - THIS IS WHERE THE FIX IS NEEDED
    class VannaLlmInterface(VannaBase):
        def __init__(self, config=None):
            VannaBase.__init__(self, config=config)
            self.model = llm_model_name
            # Store connection details needed for the API call
            self.base_url = local_server_url
            self.api_key = placeholder_api_key

        def system_message(self, message: str) -> dict: return {"role": "system", "content": message}
        def user_message(self, message: str) -> dict: return {"role": "user", "content": message}
        def assistant_message(self, message: str) -> dict: return {"role": "assistant", "content": message}

        # --- FIX: Implement actual LLM call in submit_prompt ---
        # --- FIX: Correctly parse response.choices LIST ---
        def submit_prompt(self, prompt, **kwargs) -> str:
            # ... (previous code) ...
            try:
                # ... (API call) ...
                response = client.chat.completions.create(...)

                llm_response_content = None
                # --- CORRECTED ACCESS TO choices[0] ---
                if response.choices and len(response.choices) > 0:
                    # Access the FIRST element (index 0) of the choices list
                    first_choice = response.choices[0] # <== FIX: Access the first choice object

                    # Now check attributes on the actual choice object
                    if hasattr(first_choice, 'message'):
                        message_obj = first_choice.message
                        if message_obj and hasattr(message_obj, 'content'):
                            llm_response_content = message_obj.content
                    # Optional: Handle older/different API structures if necessary, e.g., .text
                    # elif hasattr(first_choice, 'text'):
                    #     llm_response_content = first_choice.text

                # --- End Correction ---

                if llm_response_content:
                    print(f"[MyVanna.submit_prompt] LLM call successful (took {time.time() - start_llm_call:.2f}s).")
                    return llm_response_content.strip()
                else:
                    error_msg = "Error: LLM response missing content or choices structure invalid."
                    # Log the structure of the response to help debug if the issue persists
                    print(f"[MyVanna.submit_prompt] {error_msg} Raw Response: {response}")
                    return error_msg
            # ... (exception handling) ...

        # --- End Fix ---


            except openai.APIConnectionError as e:
                error_msg = f"Error: API Connection Error - {e}"
                print(f"[MyVanna.submit_prompt] {error_msg}")
                return error_msg
            except Exception as e:
                error_msg = f"Error during LLM call in submit_prompt: {type(e).__name__} - {e}"
                print(f"[MyVanna.submit_prompt] {error_msg}")
                traceback.print_exc() # Print full traceback for unexpected errors
                return error_msg
        # --- End Fix ---

    # Define Combined Vanna Class (inherits the fixed submit_prompt)
    class MyVanna(ChromaDB_VectorStore, VannaLlmInterface):
        def __init__(self, config=None):
            print("[MyVanna Init] Initializing combined Vanna class for Tool...")
            ChromaDB_VectorStore.__init__(self, config=config)
            VannaLlmInterface.__init__(self, config=config)

    print("Instantiating Vanna for Tool...")
    vn = MyVanna(config=None) # Instantiate globally

    print(f"Connecting Vanna to {db_type} database '{mysql_dbname}'...")
    if db_type == "mysql":
        vn.connect_to_mysql(host=mysql_host, port=mysql_port, user=mysql_user, password=mysql_password, dbname=mysql_dbname)
    else:
        raise ValueError(f"Database type '{db_type}' not configured.")
    print("Vanna database connection successful.")

    # --- Vanna Training ---
    print("\n--- Vanna Training ---")
    # Set train_vanna = True ONLY when you need to update Vanna's knowledge
    train_vanna = True # Default to False for normal operation
    if train_vanna:
        print("Running Vanna training...")
        try:
            # Fetch schema info
            print(f"Fetching schema for database: '{mysql_dbname}'")
            # --- FIX: Use explicit DB name with quotes ---
            sql_query = f"SELECT * FROM INFORMATION_SCHEMA.COLUMNS where TABLE_SCHEMA = '{mysql_dbname}'"
            # --- End Fix ---
            print(f"Executing SQL: {sql_query}")
            df_information_schema = vn.run_sql(sql_query)

            # --- START DEBUG PRINTS ---
            print("\n--- Debug: Information Schema DataFrame ---")
            if isinstance(df_information_schema, pd.DataFrame):
                 print(df_information_schema.head())
                 print(f"Shape: {df_information_schema.shape}")
                 if df_information_schema.empty:
                     print("ERROR: DataFrame is EMPTY. Check DB name, table existence, and permissions.")
            # --- FIX: Corrected indentation for else ---
            else:
                 print(f"ERROR: vn.run_sql did not return a DataFrame. Type: {type(df_information_schema)}")
                 print(f"Value: {df_information_schema}")
            # --- END FIX ---
            # --- END DEBUG PRINTS ---

            # Proceed only if DataFrame is not empty
            if isinstance(df_information_schema, pd.DataFrame) and not df_information_schema.empty:
                print("\nGenerating training plan...")
                plan = vn.get_training_plan_generic(df_information_schema)

                # --- START DEBUG PRINTS ---
                print("\n--- Debug: Generated Training Plan ---")
                print(f"Type: {type(plan)}")
                if isinstance(plan, list):
                     print(f"Length: {len(plan)}")
                     # print(f"First 5 elements: {plan[:5]}") # Optional verbose
                     if not plan: print("ERROR: Training plan is an EMPTY list.")
                else:
                     print(f"Value: {plan}")
                     if not plan: print("ERROR: Training plan is None or empty.")
                # --- END DEBUG PRINTS ---

                # Proceed only if plan seems valid
                if plan:
                    print("\nExecuting vn.train(plan=plan)...")
                    vn.train(plan=plan)
                    print("Schema training portion completed.")
                    
                    # === ADD DOCUMENTATION AND SQL EXAMPLES HERE ===
                    print("Adding documentation examples...")
                    # Examples for descriptive questions
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
                                    
                    print("Documentation and SQL examples added.")
                    # ===============================================
                else: print("Skipping vn.train because the plan is empty or invalid.")
            else: print("Skipping plan generation and training because schema DataFrame is empty or invalid.")

            print("\nVanna training process finished (or skipped relevant parts).")

        except Exception as train_e:
            print(f"\nError during Vanna training: {type(train_e).__name__} - {train_e}")
            traceback.print_exc()

    else: print("Skipping Vanna training (train_vanna set to False).")

except Exception as setup_e:
    print(f"\n--- Error during Vanna Setup ---")
    print(f"{type(setup_e).__name__}: {setup_e}")
    print("Cannot proceed without Vanna instance. Exiting.")
    traceback.print_exc()
    sys.exit(1)

# --- Initialize LangChain LLM ---
try:
    llm = ChatOpenAI(
        model=llm_model_name,
        openai_api_base=local_server_url,
        api_key=placeholder_api_key,
        temperature=0.1,
        max_tokens=2048,
    ) # Fixed missing parenthesis
    
    print("\n[LLM Test] Testing LangChain LLM connection...")
    llm.invoke("Hello!") # Simple test
    print("[LLM Test] LangChain LLM connection successful.")
except Exception as llm_e:
    print(f"\n--- Error initializing or testing LangChain LLM ---")
    print(f"{type(llm_e).__name__}: {llm_e}")
    print("Check if the llama-cpp-python server is running and accessible.")
    sys.exit(1)

# --- Define SQL Validation Function ---
def is_valid_sql(query: str) -> bool:
    """Check if a string looks like valid SQL."""
    if not query:
        return False
        
    # Basic check for SQL keywords at the beginning
    sql_starters = ["SELECT", "INSERT", "UPDATE", "DELETE", "WITH", "CREATE", "ALTER", "DROP", "MERGE"]
    
    # Remove any leading whitespace and convert to uppercase for comparison
    query_start = query.lstrip().upper()
    
    # Check if the query starts with any SQL keywords
    for starter in sql_starters:
        if query_start.startswith(starter):
            return True
    
    return False

# --- Define Vanna Tool for LangGraph ---
@tool
def vanna_query_tool(question: str) -> Dict[str, Any]:
    """
    Uses Vanna.ai to generate and execute a SQL query based on a natural language question.
    Returns both the generated SQL (if any) and the execution results.
    """
    global vn
    print(f"\n--- Calling Vanna Tool ---")
    # --- FIX: Use the original question directly ---
    print(f"Input Question for Vanna: {question}")
    # --- REMOVED question modification lines ---
    # sql_generation_question = f"Generate a SQL query to answer this question: {question}"
    # print(f"Modified Question for SQL Generation: {sql_generation_question}")
    # --- End Fix ---

    if vn is None: return {"sql_query": None, "results": "Error: Vanna instance not available."}

    sql_query = None
    results_str = "Error: Processing failed."
    df_result = None

    try:
        # 1. Generate SQL using the ORIGINAL question
        # --- FIX: Use 'question' variable here ---
        sql_query_raw = vn.generate_sql(question=question)
        # --- End Fix ---

        if not sql_query_raw or (isinstance(sql_query_raw, str) and sql_query_raw.startswith("Error:")):
            print(f"Vanna failed to generate SQL: {sql_query_raw}")
            results_str = sql_query_raw or "Error: Vanna did not generate SQL."
            sql_query = None
        else:
            sql_query = sql_query_raw.strip().removeprefix("``````").removesuffix("```")
            print(f"Generated SQL attempt: {sql_query}")

            if not is_valid_sql(sql_query):
                print(f"Invalid SQL detected by basic check: {sql_query}")
                results_str = f"Error: Vanna generated invalid/non-SQL text: {sql_query}"
                sql_query = None
            else:
                print("Generated text looks like SQL. Proceeding to execution...")
                try:
                    df_result = vn.run_sql(sql=sql_query)
                    print(f"SQL Execution finished. Result type: {type(df_result)}")
                    if isinstance(df_result, pd.DataFrame):
                        df_result_filled = df_result.fillna('') # Handle None before markdown
                        if df_result_filled.empty: results_str = "Query executed successfully, but returned no results."
                        elif len(df_result_filled) > 30: results_str = f"Query executed successfully. Showing first 30 rows:\n{df_result_filled.head(30).to_markdown(index=False)}\n... (truncated, {len(df_result)} rows total)"
                        else: results_str = f"Query executed successfully. Results:\n{df_result_filled.to_markdown(index=False)}"
                    elif df_result is not None: results_str = f"Query executed successfully. Result: {str(df_result)}"
                    else: results_str = "Query executed successfully, but the result was None."
                except Exception as exec_e:
                    print(f"Error executing SQL: {type(exec_e).__name__} - {exec_e}")
                    results_str = f"Error executing SQL: {type(exec_e).__name__} - {exec_e}\nFailed SQL: {sql_query}"
    except Exception as gen_e:
        print(f"Error during Vanna SQL generation step: {type(gen_e).__name__} - {gen_e}")
        results_str = f"Error during Vanna SQL generation step: {type(gen_e).__name__} - {gen_e}"
        sql_query = None

    if results_str == "Error: Processing failed." and sql_query is None:
         results_str = "Error: Failed to generate or execute SQL query."

    return {"sql_query": sql_query, "results": results_str}

# --- Prepare Tools for LangGraph ---
tools = [vanna_query_tool]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)

# --- Pattern-based Classification Function ---
def pattern_based_classification(question: str) -> str:
    """Use regex patterns to classify questions."""
    question = question.lower()
    
    # Descriptive patterns
    descriptive_patterns = [
        r"^what is", r"^how many", r"^list", r"^show", r"^tell me", 
        r"^find", r"^count", r"^where", r"^when", r"^who"
    ]
    
    # Judgment patterns
    judgment_patterns = [
        r"^can", r"^will", r"^is it possible", r"^why (was|did|is)", 
        r"^what caused", r"^is this", r"evaluate", r"assess",r"analyse"
    ]
    
    # Advice patterns
    advice_patterns = [
        r"^how (do|can|should) (i|we)", r"^what should", r"improve", 
        r"optimize", r"recommend", r"suggest", r"focus on"
    ]
    
    # Check patterns
    for pattern in descriptive_patterns:
        if re.search(pattern, question):
            return "descriptive"
    
    for pattern in judgment_patterns:
        if re.search(pattern, question):
            return "judgment"
    
    for pattern in advice_patterns:
        if re.search(pattern, question):
            return "advice"
    
    # Default to advice as the most complex type
    return "advice"

# --- LangGraph State Definition ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    question_type: str  # Added to track question type

# --- Add Question Classification Function ---
def classify_question(question: str) -> str:
    """
    Classifies a question using both LLM and pattern matching.
    """
    # Get LLM classification
    classification_prompt = [
        HumanMessage(content=f"""
        Classify the following question into EXACTLY ONE of these categories:
        
        1. DESCRIPTIVE: Questions asking for factual information or data directly available in the database. 
           Examples: "What is the total sales till now?", "How many orders were placed last month?", "List all customers in Texas."
        
        2. JUDGMENT: Questions requiring evaluation or assessment based on data.
           Examples: "Can order #123 meet its due date?", "Why was order #456 delivered late?", "Is this supplier reliable?"
        
        3. ADVICE: Questions asking for recommendations or guidance on actions to take.
           Examples: "How do I improve delivery of part X?", "Which suppliers should we focus on improving?", "What should I do about declining sales?"
        
        Question to classify: "{question}"
        
        RESPOND WITH ONLY ONE WORD - either "descriptive", "judgment", or "advice". 
        DO NOT include any explanation, reasoning, or additional text in your response.
        """)
    ]
    
    response = llm.invoke(classification_prompt)
    response_text = response.content.strip().lower()
    
    # Extract just the classification word using regex
    classification_match = re.search(r'\b(descriptive|judgment|advice)\b', response_text)
    if classification_match:
        llm_classification = classification_match.group(0)
        print(f"LLM classified question as: {llm_classification}")
    else:
        llm_classification = None
        print(f"LLM classification failed, got: {response_text}")
    
    # Get pattern-based classification
    pattern_classification = pattern_based_classification(question)
    print(f"Pattern classified question as: {pattern_classification}")
    
    # Determine final classification
    if llm_classification in ["descriptive", "judgment", "advice"]:
        final_classification = llm_classification
    else:
        final_classification = pattern_classification
    
    print(f"Final classification: {final_classification}")
    return final_classification

# --- LangGraph Nodes ---
def classifier_node(state: AgentState) -> AgentState:
    """Classifies the question and adds the classification to the state."""
    print("\n--- Classifier Node ---")
    
    # Get messages from state with error checking
    messages = state.get('messages', [])
    
    # Debug info
    print(f"[DEBUG] Messages type: {type(messages)}, count: {len(messages) if isinstance(messages, list) else 'N/A'}")
    
    # Extract question with proper error handling
    if messages and len(messages) > 0 and hasattr(messages[0], 'content'):
        question = messages[0].content
        print(f"Processing question: {question}")
    else:
        question = "No question found"
        print("Warning: Could not extract question from messages")
    
    # Use the hybrid classification approach
    question_type = classify_question(question)
    
    # Return state with classification added
    return {"messages": messages, "question_type": question_type}

def descriptive_node(state: AgentState) -> AgentState:
    """Handles descriptive questions by directly using Vanna to query the database."""
    print("\n--- Descriptive Question Node ---")
    messages = state['messages']
    question_type = state.get('question_type', 'descriptive')  # Preserve type
    
    # FIX: Access the content of the first message in the list
    if messages and len(messages) > 0 and hasattr(messages[0], 'content'):
        question = messages[0].content
    else:
        question = "No question found"
        print("Warning: Could not extract question from messages")
    
    # Always use Vanna tool for descriptive questions
    print("Descriptive question detected - Using Vanna directly")
    vanna_result = vanna_query_tool(question)
    
    # Format the response
    if vanna_result["sql_query"]:
        response_content = f"""
        I found the following information from the database:
        
        SQL Query: {vanna_result["sql_query"]}
        
        Results: {vanna_result["results"]}
        """
    else:
        response_content = f"I couldn't find the information in the database. {vanna_result['results']}"
    
    # Return complete state with question_type preserved
    return {"messages": [AIMessage(content=response_content)], "question_type": question_type}

def judgment_node(state: AgentState) -> AgentState:
    """Handles judgment questions by using Vanna for data and then analyzing results."""
    print("\n--- Judgment Question Node ---")
    messages = state['messages']
    question_type = state.get('question_type', 'judgment')  # Preserve type
    
    # For judgment questions, we still want to use the agent's decision-making,
    # but with specific instructions to use Vanna first
    judgment_system_message = """
    This question requires judgment based on data. 
    First, retrieve relevant data using the vanna_query_tool.
    Then analyze the data and provide your judgment with reasoning.
    """
    
    # Add the system message to guide the agent
    # Extract the user's question from the first message in the list
    user_question = messages[0].content if (messages and len(messages) > 0) else "No question found"
    new_messages = [SystemMessage(content=judgment_system_message), HumanMessage(content=user_question)]
    
    # Use the standard agent node but with the enhanced messages
    response = llm_with_tools.invoke(new_messages)
    
    # Return state with question_type preserved
    return {"messages": [response], "question_type": question_type}

def advice_node(state: AgentState) -> AgentState:
    """Handles advice/suggestion questions with optional data retrieval."""
    print("\n--- Advice Question Node ---")
    messages = state['messages']
    question_type = state.get('question_type', 'advice')  # Preserve type
    
    # For advice questions, we want to guide the agent to consider data
    # but also leverage broader knowledge
    advice_system_message = """
    This question asks for advice or suggestions. 
    Consider whether you need data from the database to provide informed advice.
    If so, use the vanna_query_tool to retrieve relevant information.
    Then provide thoughtful advice based on the data and best practices.
    """
    
    # Add the system message to guide the agent
    # Extract the user's question from the first message in the list
    user_question = messages[0].content if (messages and len(messages) > 0) else "No question found"
    new_messages = [SystemMessage(content=advice_system_message), HumanMessage(content=user_question)]
    
    # Use the standard agent node with the enhanced messages
    response = llm_with_tools.invoke(new_messages)
    
    # Return state with question_type preserved
    return {"messages": [response], "question_type": question_type}

def agent_node(state: AgentState) -> AgentState:
    """ Invokes the LLM to decide the next action or respond. """
    print("\n--- Agent Node ---")
    messages = state['messages']
    question_type = state.get('question_type', 'unknown')  # Preserve type
    
    print(f"[DEBUG] Agent node processing with question_type={question_type}")
    
    if isinstance(messages[-1], ToolMessage):
         print("Last message is ToolMessage, synthesizing final response.")
         response = llm.invoke(messages)
         return {"messages": [response], "question_type": question_type}
    
    print("Invoking LLM with tools to decide action...")
    response = llm_with_tools.invoke(messages)
    
    if not response.tool_calls:
         print("LLM decided no tool needed, responding directly.")
         return {"messages": [response], "question_type": question_type}
    else:
         print(f"LLM requested tool call(s): {response.tool_calls}")
         return {"messages": [response], "question_type": question_type}

# --- LangGraph Definition ---
print("\n--- Defining LangGraph Workflow ---")
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("classifier", classifier_node)
workflow.add_node("descriptive", descriptive_node)
workflow.add_node("judgment", judgment_node)
workflow.add_node("advice", advice_node)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

# Set entry point
workflow.set_entry_point("classifier")

# Define edges
# Route based on classification
workflow.add_conditional_edges(
    "classifier",
    lambda state: state.get("question_type", "advice"),  # Default to advice if missing
    {
        "descriptive": "descriptive",
        "judgment": "judgment",
        "advice": "advice"
    }
)

# Connect descriptive node directly to end (it handles everything internally)
workflow.add_edge("descriptive", END)

# Connect judgment and advice nodes to agent for potential tool use
workflow.add_edge("judgment", "agent")
workflow.add_edge("advice", "agent")

# Agent node conditionally routes to tools or END
workflow.add_conditional_edges(
    "agent",
    lambda state: "tools" if isinstance(state["messages"][-1], AIMessage) and state["messages"][-1].tool_calls else END,
    {"tools": "tools", END: END}
)

# Tools node always goes back to agent
workflow.add_edge("tools", "agent")

app = workflow.compile()
print("LangGraph app compiled with question type classification.")

# --- Main Interaction Loop (Using LangGraph with Streaming) ---
print("\n--- FactoryTwin AI Assistant (Agentic RAG via LangGraph) ---")
print(f"Connected to DB: {mysql_dbname}. Using LLM via: {local_server_url}")
print("Ask questions about your database. Type 'exit' to quit.")

while True:
    try:
        question = input("\nUser: ")
        if question.lower() == 'exit': break
        if not question.strip(): continue

        print("\nAgent working...")
        start_time = time.time()
        
        # Initialize state with messages only (classification will be added by classifier node)
        initial_state = {"messages": [HumanMessage(content=question)]}
        
        # Stream the execution for visibility
        final_state = None
        print("--- Agent Steps ---")
        for event in app.stream(initial_state):
            for node_name, output in event.items():
                print(f"Node: {node_name}")
                if isinstance(output, dict):
                    if "question_type" in output:
                        print(f"  Question classified as: {output['question_type']}")
                    if "messages" in output and output["messages"]:
                        last_msg = output["messages"][-1]
                        if hasattr(last_msg, "content"):
                            print(f"  Message Content (Partial): {str(last_msg.content)[:300]}...")
                        if hasattr(last_msg, "tool_calls"):
                            print(f"  Tool Calls: {last_msg.tool_calls}")
                final_state = output
        
        end_time = time.time()
        print("--- End Agent Steps ---")

        final_answer = "Agent Error: Could not determine final answer."
        if final_state and "messages" in final_state:
             final_response_message = final_state['messages'][-1]
             if isinstance(final_response_message, AIMessage): final_answer = final_response_message.content
             elif isinstance(final_response_message, HumanMessage): final_answer = final_response_message.content
             elif isinstance(final_response_message, ToolMessage): final_answer = f"Agent ended after tool use. Tool result: {final_response_message.content}"
        else: print("Warning: Could not determine final state or messages after stream.")

        print(f"\n--- Agent Response ({end_time - start_time:.2f}s) ---")
        print(final_answer)
        print("--- End of Agent Response ---")

    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as loop_e:
        print(f"\nAn error occurred in the main loop: {type(loop_e).__name__} - {loop_e}")
        print("Check server terminals for related errors.")
        traceback.print_exc()

print("\n--- End of Session ---")
