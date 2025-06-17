import time
import os
import sys
import warnings
import json # Keep for potential future use
import re

# Ignore specific warnings if they become noisy (optional)
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Environment Setup ---
# Set tokenizer parallelism env variable to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Package Imports and Checks ---
try:
    import vanna
    from vanna.base import VannaBase # Base class for custom LLM
    from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore # Local vector store
    import openai # Needed for direct client usage
    import mysql.connector # Check MySQL driver
    import pandas as pd # For displaying results
    print("Required packages imported successfully.")
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure 'vanna[chromadb]', 'openai', 'mysql-connector-python', 'cryptography', 'pandas', and 'llama-cpp-python' are installed.")
    print("Example: pip install 'vanna[chromadb]' openai mysql-connector-python cryptography pandas llama-cpp-python")
    sys.exit(1)

# --- Configuration ---
# Database Configuration (!!! IMPORTANT: UPDATE THESE !!!)
db_type = "mysql"
mysql_host = "127.0.0.1"             # Or "localhost", or specific host if MySQL is elsewhere
mysql_port = 3306                    # Default MySQL port
mysql_user = "root"     # Replace with your MySQL username
mysql_password = "Raekwon_wtc$36" # Replace with your MySQL password
mysql_dbname = "ft_database"         # Replace with the name of the database to use

# LLM Configuration
llm_model_name = 'DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf' # Match model served by server
local_server_url = 'http://127.0.0.1:8000/v1'
placeholder_api_key = "sk-no-key-required"

# --- Direct OpenAI Client Test (Verification) ---
# Confirms server reachability before Vanna setup
print("\n[TEST] Attempting direct OpenAI client connection to local server (v1.x)...")
direct_test_passed = False
try:
    direct_client = openai.OpenAI(base_url=local_server_url, api_key=placeholder_api_key)
    models = direct_client.models.list()
    print(f"[TEST] Successfully connected! Models found: {models.data}")
    direct_test_passed = True
except openai.APIConnectionError as e:
    print(f"[TEST] Failed to connect directly (Connection Error): {e}")
    print("[TEST] CRITICAL: Ensure the llama-cpp-python server is running and accessible at http://127.0.0.1:8000.")
except openai.AuthenticationError as e:
     print(f"[TEST] Failed to connect directly (Authentication Error): {e}")
     print("[TEST] ERROR: Unexpected authentication error during test. Check network/server.")
except Exception as e:
    print(f"[TEST] Failed to connect directly (Other Error): {type(e).__name__} - {e}")

if not direct_test_passed:
    print("[TEST] Direct connection test failed. Exiting. Please fix server accessibility.")
    sys.exit(1)
print("[TEST] Direct connection test complete.\n")
# --- End of Test ---

# --- Define CUSTOM LLM Interface Class for Vanna ---
class MyLocalLlm(VannaBase):
    def __init__(self, config=None):
        # Initialize base class
        VannaBase.__init__(self, config=config)
        # Store model name, client is created per-request in submit_prompt
        self.model = llm_model_name
        print("[LLM Init] MyLocalLlm base initialized.")

    # --- Implement abstract message methods ---
    def system_message(self, message: str) -> dict:
        """Formats a system message for the LLM."""
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> dict:
        """Formats a user message for the LLM."""
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> dict:
        """Formats an assistant message for the LLM."""
        return {"role": "assistant", "content": message}
    # --- End abstract methods ---

    # This is the core method Vanna calls to get the LLM response
        # This is the core method Vanna calls to get the LLM response
    # This is the core method Vanna calls to get the LLM response
    def submit_prompt(self, prompt, **kwargs) -> str:
        print(f"[LLM submit_prompt] Received prompt (type: {type(prompt)}, length: {len(prompt) if isinstance(prompt, (str, list)) else 'N/A'})")

        # --- FORCE CLIENT RE-INITIALIZATION ---
        try:
            # print(f"[LLM submit_prompt] Creating NEW OpenAI client for local server: {local_server_url}") # Less verbose
            local_client = openai.OpenAI(
                base_url=local_server_url, api_key=placeholder_api_key,
                timeout=kwargs.get('request_timeout', 180.0) )
        except Exception as client_e:
             print(f"[LLM submit_prompt] ERROR: Failed to create temporary OpenAI client: {client_e}")
             return f"Error: Failed to create LLM client: {client_e}"
        # --- END FORCE RE-INITIALIZATION ---

        if not isinstance(prompt, list):
             messages = [{"role": "user", "content": str(prompt)}]
        else:
             messages = prompt

        print(f"[LLM submit_prompt] Sending request to model '{self.model}' via '{local_client.base_url}'")
        request_start_time = time.time()
        default_stop_sequences = ["``````"]
        stop_sequences = kwargs.get('stop', default_stop_sequences)

        try:
            # Make the API call using the NEWLY created local_client
            response = local_client.chat.completions.create(
                model=self.model, messages=messages,
                temperature=kwargs.get('temperature', 0.1),
                max_tokens=kwargs.get('max_tokens', 1536),
                stop=stop_sequences, stream=False )
            request_end_time = time.time()

            # --- Enhanced Debugging and Robust Access for Response ---
            print(f"[LLM submit_prompt] Raw response object type: {type(response)}")
            # Print the raw response structure only if needed for deep debugging
            # if hasattr(response, 'model_dump_json'): print(f"[LLM submit_prompt] Raw response object (JSON): {response.model_dump_json(indent=2)}")
            # else: print(f"[LLM submit_prompt] Raw response object: {response}")

            llm_response = None # Initialize llm_response to None

            # Check choices structure carefully
            if response.choices and isinstance(response.choices, list) and len(response.choices) > 0:
                first_choice = response.choices[0]
                print(f"[LLM submit_prompt] First choice object type: {type(first_choice)}")
                # --- List attributes of the first choice object for debugging ---
                try:
                    print(f"[LLM submit_prompt] Attributes of first_choice: {dir(first_choice)}")
                except:
                    print("[LLM submit_prompt] Could not list attributes of first_choice.")
                # ---------------------------------------------------------------

                # Try standard path: .message.content
                if hasattr(first_choice, 'message') and first_choice.message is not None:
                    message_obj = first_choice.message
                    # print(f"[LLM submit_prompt] Message object type: {type(message_obj)}") # Optional debug
                    if hasattr(message_obj, 'content') and message_obj.content is not None:
                        llm_response = message_obj.content
                        print(f"[LLM submit_prompt] Extracted content via .message.content")
                    else:
                         print("[LLM submit_prompt] Debug: .message object missing 'content' or content is None.")
                else:
                     print("[LLM submit_prompt] Debug: First choice object missing 'message' attribute or message is None.")

                # Try fallback: .text (if .message.content failed)
                if llm_response is None and hasattr(first_choice, 'text') and first_choice.text is not None:
                     llm_response = first_choice.text
                     print(f"[LLM submit_prompt] Extracted content via .text attribute")

                # Add another fallback: maybe content is directly on the choice?
                if llm_response is None and hasattr(first_choice, 'content') and first_choice.content is not None:
                     llm_response = first_choice.content
                     print(f"[LLM submit_prompt] Extracted content via .content attribute")

                # Add yet another fallback: check for 'delta' although stream=False (sometimes seen)
                if llm_response is None and hasattr(first_choice, 'delta') and first_choice.delta is not None:
                    delta_obj = first_choice.delta
                    if hasattr(delta_obj, 'content') and delta_obj.content is not None:
                        llm_response = delta_obj.content
                        print(f"[LLM submit_prompt] Extracted content via .delta.content attribute")


                # If still no response extracted
                if llm_response is not None:
                    print(f"[LLM submit_prompt] Successfully extracted response (took {request_end_time - request_start_time:.2f}s, length: {len(llm_response)})")
                    return llm_response.strip() # Strip leading/trailing whitespace
                else:
                    # Log error and the problematic object
                    error_msg = "Error: Could not extract LLM text content from response choice. Inspected .message.content, .text, .content, .delta.content."
                    print(f"[LLM submit_prompt] {error_msg}")
                    print(f"[LLM submit_prompt] Failed on first_choice object: {first_choice}") # Print the object that failed
                    return error_msg # Return the specific error

            else:
                error_msg = "Error: LLM response missing 'choices', choices not a list, or choices empty."
                print(f"[LLM submit_prompt] {error_msg}")
                # Print the whole response if choices structure is wrong
                print(f"[LLM submit_prompt] Whole response object: {response}")
                return error_msg
            # --- End Enhanced Debugging ---

        except openai.APIConnectionError as e:
            print(f"[LLM submit_prompt] ERROR: Connection to LLM server failed: {e}")
            return f"Error: Could not connect to the local LLM server."
        except Exception as e:
            print(f"[LLM submit_prompt] ERROR: An unexpected error occurred during LLM call: {type(e).__name__} - {e}")
            import traceback; traceback.print_exc()
            return f"Error: An unexpected error occurred querying LLM: {e}"



# --- Define Combined Vanna Class ---
class MyVanna(ChromaDB_VectorStore, MyLocalLlm):
     def __init__(self, config=None):
        print("[MyVanna Init] Initializing combined Vanna class...")
        ChromaDB_VectorStore.__init__(self, config=config)
        print("[MyVanna Init] ChromaDB part initialized.")
        MyLocalLlm.__init__(self, config=config)
        print("[MyVanna Init] Custom LLM part initialized.")

# --- Instantiate Vanna ---
print("Instantiating Vanna using custom local LLM class...")
try:
    # config=None uses default ChromaDB path .vanna/chroma
    vn = MyVanna(config=None)
    print("Vanna instantiated successfully.")
except Exception as e:
    print(f"Error instantiating MyVanna: {e}")
    sys.exit(1)

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
    try:
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
        vn.train(documentation="PART_MST.p_code is the unique part code/number.")
        vn.train(documentation="PART_MST.p_desc provides a text description of the part.")
        vn.train(documentation="PART_MST.p_type categorizes the part (e.g., raw material, subassembly, finished good).")
        vn.train(documentation="PART_MST.procurement indicates if a part is manufactured internally ('M') or purchased ('B'). It must be either 'M' or 'B'.")
        vn.train(documentation="PART_MST.mrp_type specifies the Material Requirements Planning strategy for the part.")
        vn.train(documentation="PART_MST.sys_lt is the standard System Lead Time for the part in days.")
        vn.train(documentation="PART_MST.vend_id links to the VENDOR table, indicating the primary supplier if procurement is 'B'.")

        # --- Sales Order Header Table Documentation ---
        vn.train(documentation="Table SALES_MST represents the header information for a customer sales order.")
        vn.train(documentation="SALES_MST.order_no is the unique identifier for a sales order.")
        vn.train(documentation="SALES_MST.order_date is the date the sales order was placed.")
        vn.train(documentation="SALES_MST.cust links to the CUSTOMER table for the customer who placed the order.")

        # --- Sales Order Line Table Documentation ---
        vn.train(documentation="Table SALES_LINE contains individual line items for each sales order in SALES_MST.")
        vn.train(documentation="SALES_LINE.order_no links back to the SALES_MST table.")
        vn.train(documentation="SALES_LINE.p_code links to the PART_MST table for the specific part ordered.")
        vn.train(documentation="SALES_LINE.due_date is the requested delivery date for this line item.")
        vn.train(documentation="SALES_LINE.unit_price is the agreed price per unit for the part on this order line.")
        vn.train(documentation="SALES_LINE.qty is the number of units ordered for this part on this line item.")

        # --- Purchase Order Header Table Documentation ---
        vn.train(documentation="Table PURCHASE_MST represents the header information for a purchase order sent to a vendor.")
        vn.train(documentation="PURCHASE_MST.order_no is the unique identifier for a purchase order.")
        vn.train(documentation="PURCHASE_MST.order_date is the date the purchase order was created.")
        vn.train(documentation="PURCHASE_MST.vend links to the VENDOR table for the supplier receiving the order.")

        # --- Purchase Order Line Table Documentation ---
        vn.train(documentation="Table PURCHASE_LINE contains individual line items for each purchase order in PURCHASE_MST.")
        vn.train(documentation="PURCHASE_LINE.order_no links back to the PURCHASE_MST table.")
        vn.train(documentation="PURCHASE_LINE.part links to the PART_MST table for the specific part being purchased.")
        vn.train(documentation="PURCHASE_LINE.due_date is the expected delivery date for this purchased part.")
        vn.train(documentation="PURCHASE_LINE.unit_cost is the cost per unit for the part on this purchase order line.")
        vn.train(documentation="PURCHASE_LINE.qty is the number of units purchased. It must be a positive number.")

        # --- Job Master Table Documentation ---
        vn.train(documentation="Table JOB_MST represents production or work jobs.")
        vn.train(documentation="JOB_MST.job_no is the unique identifier for a job.")
        vn.train(documentation="JOB_MST.job_desc provides a description of the job's purpose.")
        vn.train(documentation="JOB_MST.job_date is typically the creation or start date of the job.")
        vn.train(documentation="JOB_MST.job_type categorizes the job (e.g., assembly, machining).")
        vn.train(documentation="JOB_MST.job_stat indicates the current status: 'Q' (queue), 'C' (completed), 'O' (opened), 'X' (cancelled), 'H' (on hold).")
        vn.train(documentation="JOB_MST.job_cls is the date the job was closed or completed. This should only be populated if job_stat is 'C' or 'X'.")
        vn.train(documentation="JOB_MST.part links to the PART_MST table if the job is for producing a specific part.")

        # --- Add Documentation for BOM_MST, RTG_MST, JOB_OPER, JOB_MATL, RECV_MST, RECV_LINE ---
        # Example:
        vn.train(documentation="Table BOM_MST defines the Bill of Materials, showing parent-child part relationships.")
        vn.train(documentation="BOM_MST.bom_parent links to the parent part code in PART_MST.")
        vn.train(documentation="BOM_MST.bom_child links to the child component part code in PART_MST.")
        vn.train(documentation="BOM_MST.bom_qty is the quantity of the child part needed to make one unit of the parent part.")

        # --- General Business Rules/Concepts ---
        vn.train(documentation="'Lead Time' generally refers to the time required to procure or produce a part (PART_MST.sys_lt is a default).")
        vn.train(documentation="'On-time Delivery' can be assessed by comparing SALES_LINE.due_date with actual delivery dates (potentially in another table like deliveries if it exists).")



        print("  Example: Adding Sample SQL (Add yours)...")
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
                    sql="""SELECT c.id AS customer_id, c.name AS customer_name, SUM(sl.qty * sl.unit_price) AS total_spent FROM CUSTOMER c JOIN SALES_MST s ON c.id = s.cust JOIN SALES_LINE sl ON s.order_no = sl.order_no WHERE s.order_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH) AND sl.order_stat <> 'X' -- Exclude cancelled lines GROUP BY c.id, c.name ORDER BY total_spent DESC LIMIT 5;""")
        
        vn.train(question="Parts Below Preferred Order Quantity", 
                    sql="""SELECT id,p_code,curr_stock, pref_order_qty FROM PART_MST WHERE curr_stock < pref_order_qty AND procurement <> 'S' -- Exclude Service parts ORDER BY (pref_order_qty - curr_stock) DESC; -- Order by largest deficit""")
        
        vn.train(question="Monthly Sales Revenue Trend", 
                    sql="""SELECT DATE_FORMAT(s.order_date, '%Y-%m') AS sales_month, round(SUM(sl.qty * sl.unit_price),2) AS monthly_revenue FROM SALES_MST s JOIN SALES_LINE sl ON s.order_no = sl.order_no WHERE sl.order_stat <> 'X' -- Exclude cancelled lines GROUP BY sales_month ORDER BY sales_month;""")

        vn.train(question="Direct Components for a Manufactured Part (Bill of Materials)", 
                    sql="""SELECT mp.p_code AS manufactured_part_code, cp.p_code AS component_part_code, bom.qty_req AS quantity_required, cp.unit_meas AS component_unit FROM BILL_OF_M bom JOIN PART_MST mp ON bom.m_part = mp.id  JOIN PART_MST cp ON bom.c_part = cp.id  WHERE bom.m_part = 5638;""")
        
        vn.train(question="Total Component Quantity Required for Open Production Jobs", 
                        sql="""SELECT bom.c_part AS component_part_id, p.p_code AS component_part_code, SUM(j.qty * bom.qty_req) AS total_required_for_open_jobs FROM JOB_MST j JOIN BILL_OF_M bom ON j.part = bom.m_part -- Join Job's manufactured part to BOM JOIN PART_MST p ON bom.c_part = p.id   -- Get component part details WHERE j.job_stat IN ('O', 'Q') -- Only consider Open or Queued jobs GROUP BY bom.c_part, p.p_code ORDER BY total_required_for_open_jobs DESC;""")
            
        vn.train(question="Nonconformance Rate per Part (Simplified: NC Qty / Qty Purchased)", 
                        sql="""WITH PartPurchasesReceived AS (SELECT pl.part, SUM(pl.qty) AS total_received FROM PURCHASE_LINE pl WHERE pl.order_stat = 'C' GROUP BY pl.part),PartNonconformance AS (SELECT pl.part, SUM(nc.qty) AS total_nonconforming FROM NONCONFORM_MST nc JOIN PURCHASE_MST pm ON nc.po_no = pm.order_no  JOIN PURCHASE_LINE pl ON pm.order_no = pl.order_no AND nc.job IS NULL     GROUP BY pl.part)SELECT p.id AS part_id, p.p_code, COALESCE(ppr.total_received, 0) AS total_received, COALESCE(pnc.total_nonconforming, 0) AS total_nonconforming, CASE WHEN COALESCE(ppr.total_received, 0) > 0 THEN (COALESCE(pnc.total_nonconforming, 0) / ppr.total_received) * 100 ELSE 0 END AS nonconformance_rate_percent FROM PART_MST p LEFT JOIN PartPurchasesReceived ppr ON p.id = ppr.part LEFT JOIN PartNonconformance pnc ON p.id = pnc.part WHERE p.procurement = 'B'  ORDER BY nonconformance_rate_percent DESC;""")
        
        vn.train(question="Customers Who Haven't Ordered in Over a Year", 
                        sql="""SELECT c.id, c.name, MAX(s.order_date) AS last_order_date  FROM CUSTOMER c LEFT JOIN SALES_MST s ON c.id = s.cust AND s.order_date >= DATE_SUB(CURDATE(), INTERVAL 1 YEAR) WHERE s.order_no IS NULL  GROUP BY c.id, c.name  ORDER BY c.name;""")
        
        vn.train(question="Parts Purchased from Multiple Vendors with Cost Comparison", 
                        sql="""SELECT p.id AS part_id, p.p_code, COUNT(DISTINCT vp.vendor) AS number_of_vendors, MIN(vp.unit_cost) AS min_vendor_cost, MAX(vp.unit_cost) AS max_vendor_cost, (MAX(vp.unit_cost) - MIN(vp.unit_cost)) AS cost_difference FROM PART_MST p JOIN VEND_PART vp ON p.id = vp.part GROUP BY p.id, p.p_code HAVING COUNT(DISTINCT vp.vendor) > 1  ORDER BY cost_difference DESC, p.p_code;""")
        

        print("Vanna training section complete (customize with your data).")
        # === End of custom training calls ===
    except Exception as train_e:
        print(f"Error during Vanna training: {train_e}")
else:
    print("Skipping Vanna training. Ensure metadata is already loaded or set train_vanna = True.")

# --- Define SQL Extraction Function ---
def extract_sql_from_response(response_text: str) -> str | None:
    """
    Extracts SQL query from LLM response.
    Looks for markdown code blocks (```
    then falls back to finding common SQL keywords.
    """
    if not isinstance(response_text, str):
        return None

    # 1. Try finding ```sql...```
    match_code_block = re.search(r"```sql\s*(.*?)\s*```", response_text, re.DOTALL | re.IGNORECASE)
    if match_code_block:
        print("[SQL Extract] Found SQL using ```sql block.")
        return match_code_block.group(1).strip()

    # 2. Try finding simple `````` blocks containing SQL
    # FIX: Added response_text as the second argument
    match_simple_block = re.search(r"``````", response_text, re.DOTALL | re.IGNORECASE)
    if match_simple_block:
        print("[SQL Extract] Found SQL using simple ```")
        return match_simple_block.group(1).strip()

    # 3. Fallback: Find last occurrence of SELECT or WITH (case-insensitive)
    select_pos = response_text.rfind("SELECT ")
    with_pos = response_text.rfind("WITH ")
    start_pos = -1
    if select_pos != -1 and with_pos != -1: start_pos = max(select_pos, with_pos)
    elif select_pos != -1: start_pos = select_pos
    elif with_pos != -1: start_pos = with_pos

    if start_pos != -1:
        potential_sql = response_text[start_pos:].strip()
        end_match = re.search(r"(;\s*|$)", potential_sql)
        if end_match:
            potential_sql = potential_sql[:end_match.end()].strip()
            if potential_sql.count("(") >= potential_sql.count(")"):
                print("[SQL Extract] Found SQL using fallback (last SELECT/WITH).")
                return potential_sql

    # 4. If no SQL found
    print("[SQL Extract] No SQL query patterns found in the response.")
    return None
# --- End SQL Extraction Function ---


# --- Asking Questions Loop ---
print("\n--- Vanna AI Assistant (using custom local LLM) ---")
print(f"Connected to DB: {mysql_dbname}. Using LLM via: {local_server_url}")
print("Ask questions about your database. Type 'exit' to quit.")

while True:
    try:

        question = input("\nUser: ")
        if question.lower() == 'exit':
            break
        if not question.strip():
            continue

        print("\nAgent thinking... (Calling Vanna custom LLM interface)")
        start_time = time.time()

        # 1. Get the full response (including reasoning) from the LLM via Vanna
        llm_full_response = vn.generate_sql(question=question)

        end_time = time.time()

        # 2. Print the entire response for the user to see the reasoning
        print(f"\n--- LLM Full Response ({end_time - start_time:.2f}s) ---")
        print(llm_full_response or "LLM did not provide a response.")
        print("--- End of LLM Full Response ---")

        # Check if the response indicates an error occurred during generation
        if isinstance(llm_full_response, str) and llm_full_response.startswith("Error:"):
            print("\nAn error occurred during LLM communication. Cannot proceed.")
            continue # Skip SQL extraction and execution if generation failed

        # 3. Attempt to extract only the SQL part from the full response
        extracted_sql = extract_sql_from_response(llm_full_response)

        # 4. Check if SQL extraction was successful
        if extracted_sql:
            # --- SQL EXTRACTED SUCCESSFULLY ---
            print(f"\n--- Extracted SQL ---")
            print(extracted_sql)
            print("--- End of Extracted SQL ---")

            # --- Optional: Execute the EXTRACTED SQL ---
            execute = input("\nExecute the EXTRACTED SQL query above? (y/n): ").lower().strip()
            if execute == 'y':
                try:
                    print("Executing query...")
                    exec_start_time = time.time()
                    # Execute ONLY the extracted SQL
                    df_result = vn.run_sql(sql=extracted_sql)
                    exec_end_time = time.time()
                    print(f"\nQuery Result (DataFrame) ({exec_end_time - exec_start_time:.2f}s):")
                    pd_display_rows = 20
                    with pd.option_context('display.max_rows', pd_display_rows, 'display.max_columns', None, 'display.width', 1000):
                        if len(df_result) > pd_display_rows:
                             print(df_result.head(pd_display_rows))
                             print(f"... (truncated, {len(df_result)} rows total)")
                        else:
                             print(df_result)
                except Exception as exec_e:
                    print(f"\nError executing SQL: {type(exec_e).__name__} - {exec_e}")
                    print(f"Failed SQL:\n{extracted_sql}")
            # --- End Optional Execute ---
            # --- END SUCCESSFUL SQL BLOCK ---

        else:
            # --- SQL EXTRACTION FAILED ---
            print("\nCould not extract an executable SQL query from the LLM's full response above.")
            # --- END FAILED SQL BLOCK ---

    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as loop_e:
        # Catch other potential errors during the main loop
        print(f"\nAn error occurred in the main loop: {type(loop_e).__name__} - {loop_e}")
        print("Check the llama-cpp-python server terminal for any related errors.")
        import traceback
        traceback.print_exc()

print("\n--- End of Session ---")

