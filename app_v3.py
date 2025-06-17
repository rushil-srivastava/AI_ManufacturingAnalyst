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

# --- Image URLs ---
logo_url = "https://static.wixstatic.com/media/29b997_2a8e386d077a4ae7b702b4badb265027~mv2.png/v1/fill/w_1200,h_316,al_c,q_90,usm_0.66_1.00_0.01,enc_avif,quality_auto/FactoryTwin%20Full%20Logo%20w%20Tagline%20(1)%20(1).png"
background_url = "https://static.wixstatic.com/media/29b997_8ff932375b7544ada7c8ed7e5ce09b16~mv2.jpg/v1/fill/w_3000,h_1151,al_c,q_90,enc_avif,quality_auto/29b997_8ff932375b7544ada7c8ed7e5ce09b16~mv2.jpg"

# --- CSS Styling ---
st.markdown(f"""
<style>
    div.block-container:first-of-type {{
        padding-top: 0rem;
        padding-bottom: 0rem;
    }}

    .banner-container {{
        width: 100%;
        background-image: url('{background_url}');
        background-size: cover;
        background-position: center;
        padding: 2rem 1rem;
        margin-bottom: 1rem;
        text-align: center;
    }}

    .banner-container img {{
        max-width: 80%;
        height: auto;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }}

    h1 {{
        color: black;
        text-align: left;
        padding-left: 1rem;
    }}
</style>
""", unsafe_allow_html=True)

# --- Display Banner ---
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
    
    # Check if this is a retry attempt
    if st.session_state.get('retry_count', 0) > 0:
        retry_context = f"""
        Previous attempt failed with error: {st.session_state.previous_error if st.session_state.previous_error else 'No specific error'}
        Previous SQL: {st.session_state.previous_sql}
        Please provide a different approach to answer: {st.session_state.question}
        """
        st.session_state.new_question = retry_context
    
    with st.spinner("Agent thinking... Generating analysis..."):
        # For suggestion questions, directly get recommendations
        if st.session_state.category == "suggestion":
            # More concise prompt for suggestions
            st.session_state.new_question = f"""
            Question: {st.session_state.question}
            Provide 3-5 key recommendations. Keep each recommendation brief and actionable.
            Format: 
            1. Recommendation 1
            2. Recommendation 2
            etc.
            """
            
            try:
                st.session_state.llm_response = vn.generate_contextual_response(st.session_state.new_question)
                st.write("**Recommendations:**")
                st.markdown(st.session_state.llm_response)
                
                # Add options to save or retry
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Save Response"):
                        st.success("Response saved successfully!")
                with col2:
                    if st.button("Try Different Approach"):
                        st.session_state.retry_count = st.session_state.get('retry_count', 0) + 1
                        set_step("generate_sql")
                        st.rerun()
                with col3:
                    if st.button("New Question"):
                        reset_app()
                        st.rerun()
            except Exception as e:
                if "context_length_exceeded" in str(e):
                    st.error("Response too long. Trying with a more concise approach...")
                    # Try with an even more concise prompt
                    st.session_state.new_question = f"""
                    Question: {st.session_state.question}
                    Provide 3 key recommendations. Be very concise.
                    """
                    st.session_state.llm_response = vn.generate_contextual_response(st.session_state.new_question)
                    st.write("**Recommendations:**")
                    st.markdown(st.session_state.llm_response)
                else:
                    st.error(f"Error generating response: {str(e)}")
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
            # First try to find similar questions in the vector database
            with st.spinner("Searching for similar questions..."):
                try:
                    # Search for similar questions
                    similar_questions = vn.get_similar_questions(st.session_state.question)
                    
                    if similar_questions and len(similar_questions) > 0:
                        st.write("**Found similar questions in database:**")
                        for idx, q in enumerate(similar_questions[:3], 1):  # Show top 3 matches
                            st.write(f"{idx}. {q}")
                        
                        # Get the SQL for the most similar question
                        most_similar_sql = vn.get_sql_for_question(similar_questions[0])
                        
                        if most_similar_sql:
                            st.write("**Using SQL from similar question:**")
                            st.code(most_similar_sql, language="sql")
                            
                            # Ask user if they want to use this SQL
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("Use This SQL"):
                                    st.session_state.extracted_sql = most_similar_sql
                                    set_step("confirm_execution")
                                    st.rerun()
                            with col2:
                                if st.button("Generate New SQL"):
                                    st.session_state.llm_response = generate_complex_sql(st.session_state.new_question)
                                    st.write("**LLM Response:**")
                                    st.markdown(st.session_state.llm_response)
                                    try:
                                        extracted_sql = extract_sql_from_response(st.session_state.llm_response)
                                        if extracted_sql:
                                            st.session_state.extracted_sql = extracted_sql
                                            st.write("**Extracted SQL:**")
                                            st.code(extracted_sql, language="sql")
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
                    else:
                        # If no similar questions found, generate new SQL
                        st.session_state.llm_response = generate_complex_sql(st.session_state.new_question)
                        st.write("**LLM Response:**")
                        st.markdown(st.session_state.llm_response)
                        try:
                            extracted_sql = extract_sql_from_response(st.session_state.llm_response)
                            if extracted_sql:
                                st.session_state.extracted_sql = extracted_sql
                                st.write("**Extracted SQL:**")
                                st.code(extracted_sql, language="sql")
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
                except Exception as e:
                    st.error(f"Error searching for similar questions: {str(e)}")
                    # Fallback to generating new SQL
                    st.session_state.llm_response = generate_complex_sql(st.session_state.new_question)
                    st.write("**LLM Response:**")
                    st.markdown(st.session_state.llm_response)
                    try:
                        extracted_sql = extract_sql_from_response(st.session_state.llm_response)
                        if extracted_sql:
                            st.session_state.extracted_sql = extracted_sql
                            st.write("**Extracted SQL:**")
                            st.code(extracted_sql, language="sql")
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

# Step 3: Confirm SQL execution
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
            st.dataframe(df_result)
            st.session_state.explanation = analyze_and_explain(st.session_state.question, df_result, "judgement")
            st.write("**Judgement Analysis:**")
            st.markdown(st.session_state.explanation)
            
        elif st.session_state.category == "suggestion":
            st.write("**Query Results:**")
            st.dataframe(df_result)
            st.session_state.explanation = analyze_and_explain(st.session_state.question, df_result, "suggestion")
            st.write("**Recommendations:**")
            st.markdown(st.session_state.explanation)
            
        else:  # descriptive
            st.write("**Query Results:**")
            st.dataframe(df_result)
            st.session_state.explanation = analyze_and_explain(st.session_state.question, df_result, "descriptive")
            st.write("**Summary:**")
            st.markdown(st.session_state.explanation)

        # Add options to save or retry
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Save Query"):
                vn.train(question=st.session_state.question, sql=st.session_state.extracted_sql)
                st.success("Query saved successfully!")
        with col2:
            if st.button("Try Different Approach"):
                # Store the current state for retry
                st.session_state.previous_sql = st.session_state.extracted_sql
                st.session_state.previous_error = None
                st.session_state.retry_count = st.session_state.get('retry_count', 0) + 1
                set_step("generate_sql")
                st.rerun()
        with col3:
            if st.button("New Question"):
                reset_app()
                st.rerun()

    except Exception as e:
        st.error(f"Error executing SQL: {str(e)}")
        # Store error information for retry
        st.session_state.previous_sql = st.session_state.extracted_sql
        st.session_state.previous_error = str(e)
        st.session_state.retry_count = st.session_state.get('retry_count', 0) + 1
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Try Again"):
                set_step("generate_sql")
                st.rerun()
        with col2:
            if st.button("Cancel"):
                reset_app()
                st.rerun() 