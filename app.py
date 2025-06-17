import streamlit as st
import pandas as pd
import time
from datetime import datetime

# Import your own code (assumes mistral_v2.py is in the same directory or installed as a module)
from mistral_v2 import (
    vn,  # Your MyVanna object, already connected to DB and trained
    extract_sql_from_response,
    generate_complex_sql,
    analyze_and_explain,
    current_date,
)

st.set_page_config(page_title="Manufacturing SQL Agent", layout="wide")
# --- 1. Set Page Config (MUST be the first Streamlit command) ---
# st.set_page_config(layout="wide") # Use wide mode for full width

# --- Image URLs ---
logo_url = "https://static.wixstatic.com/media/29b997_2a8e386d077a4ae7b702b4badb265027~mv2.png/v1/fill/w_1200,h_316,al_c,q_90,usm_0.66_1.00_0.01,enc_avif,quality_auto/FactoryTwin%20Full%20Logo%20w%20Tagline%20(1)%20(1).png"
background_url = "https://static.wixstatic.com/media/29b997_8ff932375b7544ada7c8ed7e5ce09b16~mv2.jpg/v1/fill/w_3000,h_1151,al_c,q_90,enc_avif,quality_auto/29b997_8ff932375b7544ada7c8ed7e5ce09b16~mv2.jpg"

# --- 2. Inject CSS for Banner Styling and Padding Removal ---
st.markdown(f"""
<style>
    /* Target the Streamlit block container */
    /* Adjust the 'nth-of-type' if needed, usually 1 or 2 */
    div.block-container:first-of-type {{
        padding-top: 0rem; /* Remove top padding */
        padding-bottom: 0rem; /* Optional: Remove bottom padding if needed */
    }}

    .banner-container {{
        width: 100%; /* Make container full width */
        background-image: url('{background_url}');
        background-size: cover;
        background-position: center;
        padding: 2rem 1rem; /* Add vertical padding, adjust as needed */
        margin-bottom: 1rem; /* Space below the banner */
        text-align: center; /* Center the image inside */
        /* border-radius: 10px; /* Optional: Rounded corners */
    }}

    .banner-container img {{
        max-width: 80%; /* Control logo size, adjust percentage */
        height: auto;
        display: block; /* Ensures block behavior */
        margin-left: auto; /* Center the image */
        margin-right: auto; /* Center the image */
        /* Optional: add some effect like a drop shadow */
        /* filter: drop-shadow(0 4px 8px rgba(0,0,0,0.3)); */
    }}

    /* Optional: Adjust padding for the rest of the content below banner */
    /* div.block-container:nth-of-type(2) {{
         padding-top: 2rem;
    }} */

    /* Style for the title */
    h1 {{
        color: black;
        text-align: left; /* Or 'center' if you prefer */
        padding-left: 1rem; /* Align with content padding */
    }}
</style>
""", unsafe_allow_html=True)

# --- 3. Display the Banner (comes BEFORE the title) ---
st.markdown(f"""
<div class="banner-container">
    <img src="{logo_url}" alt="FactoryTwin Logo">
</div>
""", unsafe_allow_html=True)
st.title("AI Agent")
st.subheader("Ask your manufacturing data questions and get SQL queries and insights!")

# --- Initialize Session State Variables ---
if "question" not in st.session_state:
    st.session_state.question = ""
if "new_question" not in st.session_state:
    st.session_state.new_question = ""
if "category" not in st.session_state:
    st.session_state.category = ""
if "llm_response" not in st.session_state:
    st.session_state.llm_response = ""
if "extracted_sql" not in st.session_state:
    st.session_state.extracted_sql = ""
if "df_result" not in st.session_state:
    st.session_state.df_result = None
if "explanation" not in st.session_state:
    st.session_state.explanation = ""
if "current_step" not in st.session_state:
    st.session_state.current_step = "input_question"
if "execute_sql" not in st.session_state:
    st.session_state.execute_sql = False
if "error_message" not in st.session_state:
    st.session_state.error_message = ""
if "plotly_figure" not in st.session_state:
    st.session_state.plotly_figure = None

# --- Helper Functions ---
def reset_app():
    st.session_state.question = ""
    st.session_state.new_question = ""
    st.session_state.category = ""
    st.session_state.llm_response = ""
    st.session_state.extracted_sql = ""
    st.session_state.df_result = None
    st.session_state.explanation = ""
    st.session_state.current_step = "input_question"
    st.session_state.execute_sql = False
    st.session_state.error_message = ""
    st.session_state.plotly_figure = None

def set_step(step_name):
    st.session_state.current_step = step_name

# --- Main App Logic ---
# Step 1: Get user question
if st.session_state.current_step == "input_question":
    with st.form("question_form"):
        user_question = st.text_input("Ask a manufacturing data question (type 'exit' to quit):", key="input_question_field")
        submit = st.form_submit_button("Submit")
    
    if submit:
        if user_question.lower().strip() == "exit":
            st.success("Session ended.")
            st.stop()
        else:
            st.session_state.question = user_question
            st.session_state.new_question = f"Remember Today's date for date related queries {current_date}. Answer this question: {user_question}"
            set_step("generate_sql")
            st.rerun()

# Step 2: Generate SQL
elif st.session_state.current_step == "generate_sql":
    st.info(f"Processing question: {st.session_state.question}")
    
    with st.spinner("Classifying question..."):
        st.session_state.category = vn.classify_question(st.session_state.new_question)
    
    st.write(f"**Classified Category:** `{st.session_state.category.upper()}`")
    
    with st.spinner("Agent thinking... Generating SQL and analysis..."):
        if st.session_state.category.lower() == "suggestion":
            st.session_state.llm_response = vn.generate_contextual_response(st.session_state.new_question)
        else:    
            st.session_state.llm_response = generate_complex_sql(st.session_state.new_question)
    
    st.write("**LLM Response:**")
    st.markdown(st.session_state.llm_response)

    try:
        extracted_sql = extract_sql_from_response(st.session_state.llm_response)
        if not extracted_sql:
            st.error("Could not extract SQL from the response.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Try Again"):
                    set_step("generate_sql")
                    st.rerun()
            with col2:
                if st.button("Cancel"):
                    reset_app()
                    st.rerun()
        
        else:
            st.session_state.extracted_sql = extracted_sql
            st.write("**Extracted SQL:**")
            st.code(extracted_sql, language="sql")
            
            if st.session_state.category == "suggestion":
                set_step("handle_suggestion")
            else:
                set_step("confirm_execution")
            st.rerun()
    except Exception as e:
        st.error(f"Error extracting SQL: {str(e)}")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Try Again"):
                set_step("generate_sql")
                st.rerun()
        with col2:
            if st.button("Cancel"):
                reset_app()
                st.rerun()

# Step 3a: Handle suggestion category
elif st.session_state.current_step == "handle_suggestion":
    st.write("**Suggestion from Agent:**")

    # Ensure the full LLM response is displayed using st.markdown
    # This assumes st.session_state.llm_response contains the full text
    # including SQL (often in a markdown code block) and the suggestions.
    if st.session_state.llm_response:
        st.text_area("Suggestion and SQL Response", value=st.session_state.llm_response, height=400, disabled=True)
    else:
        st.warning("No suggestion content found in the LLM response.")

    # Optional: Display the extracted SQL separately if needed for clarity,
    # but the full response above should already contain it.
    # if st.session_state.extracted_sql:
    #     st.write("**Extracted SQL (for reference):**")
    #     st.code(st.session_state.extracted_sql, language="sql")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("I'm Satisfied"):
            reset_app()
            st.rerun()
    with col2:
        if st.button("Rethink Suggestion"):
            # Prepare to regenerate, perhaps refining the prompt if needed
            st.session_state.new_question = f"Please provide a different suggestion for: {st.session_state.question}" # Example refinement
            set_step("generate_sql")
            st.rerun()

# Step 3b: Confirm SQL execution for non-suggestion categories
elif st.session_state.current_step == "confirm_execution":
    st.write("**Ready to execute the extracted SQL against the database.**")
    st.code(st.session_state.extracted_sql, language="sql")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button(f"Execute {st.session_state.category.upper()} SQL Query"):
            st.session_state.execute_sql = True
            set_step("execute_sql")
            st.rerun()
    with col2:
        if st.button("Cancel"):
            reset_app()
            st.rerun()

# Step 4: Execute SQL and show results
elif st.session_state.current_step == "execute_sql":
    st.write("**Executing SQL query...**")
    
    try:
        with st.spinner("Running query..."):
            exec_start_time = time.time()
            df_result = vn.run_sql(sql=st.session_state.extracted_sql)
            st.session_state.df_result = df_result
        
        st.success(f"Query executed in {time.time() - exec_start_time:.2f} seconds.")
        
        # Display results based on category
        if st.session_state.category == "judgement":
            st.write("**Analytical Insights:**")
            # st.dataframe(df_result.describe())
        
        st.write("**Query Results:**")
        if len(df_result) > 20:
            st.dataframe(df_result.head(20))
            st.info(f"... (truncated, {len(df_result)} rows total)")
        else:
            st.dataframe(df_result)
        
        # Generate and display visualization
        with st.spinner("Generating visualization..."):
            try:
                st.write("**Data Visualization:**")
                plotly_code = vn.generate_plotly_code(
                    question=st.session_state.question,
                    sql=st.session_state.extracted_sql,
                    df=df_result
                )
                if plotly_code:
                    fig = vn.get_plotly_figure(plotly_code=plotly_code, df=df_result)
                    st.plotly_chart(fig, use_container_width=True)
                    st.session_state.plotly_figure = fig
                else:
                    st.info("No visualization could be generated for this query.")
            except Exception as viz_error:
                st.warning(f"Could not generate visualization: {str(viz_error)}")
        
        # Generate explanation
        with st.spinner("Analyzing results..."):
            explanation = analyze_and_explain(st.session_state.question, df_result, st.session_state.category)
            st.session_state.explanation = explanation
        
        st.write(f"**{st.session_state.category.upper()} EXPLANATION:**")
        st.write(explanation)
        
        set_step("feedback")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error executing SQL: {type(e).__name__} - {str(e)}")
        st.session_state.error_message = f"The query: {st.session_state.extracted_sql} is giving this error: {type(e).__name__} - {str(e)}"
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Rethink Query"):
                st.session_state.new_question = f"{st.session_state.error_message} Resolve the error and provide a better response for this question: {st.session_state.question}"
                set_step("generate_sql")
                st.rerun()
        with col2:
            if st.button("Start Over"):
                reset_app()
                st.rerun()

# Step 5: Get feedback on results
elif st.session_state.current_step == "feedback":
    st.write("**Results and Explanation:**")
    
    # Show a condensed version of the results
    if st.session_state.df_result is not None:
        if len(st.session_state.df_result) > 5:
            st.dataframe(st.session_state.df_result.head(5))
            st.info(f"... (showing 5 of {len(st.session_state.df_result)} rows)")
        else:
            st.dataframe(st.session_state.df_result)
    
    # Show visualization if available
    if st.session_state.plotly_figure is not None:
        st.plotly_chart(st.session_state.plotly_figure, use_container_width=True)
    
    st.write(st.session_state.explanation)
    
    st.write("**Are you satisfied with these results?**")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Yes, I'm Satisfied"):
            set_step("save_query")
            st.rerun()
    
    with col2:
        if st.button("No, Rethink Query"):
            st.session_state.new_question = f"Please improve your response to this question: {st.session_state.question}"
            set_step("generate_sql")
            st.rerun()

# Step 6: Save query option
elif st.session_state.current_step == "save_query":
    st.success("Query and explanation ready.")
    st.write("**Do you want to save this SQL query for future use?**")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes, Save Query"):
            try:
                vn.train(question=st.session_state.question, sql=st.session_state.extracted_sql)
                st.success("Query saved to vector database for future use.")
                reset_app()
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save query: {e}")
    
    with col2:
        if st.button("No, Don't Save"):
            reset_app()
            st.rerun()

# --- Debug Information ---
with st.expander("Show Debug Info"):
    st.write("Current Date:", current_date)
    st.write("Current Step:", st.session_state.current_step)
    st.write("Question:", st.session_state.question)
    st.write("Category:", st.session_state.category)
    st.write("Has Results:", st.session_state.df_result is not None)
    st.write("Has Visualization:", st.session_state.plotly_figure is not None)
