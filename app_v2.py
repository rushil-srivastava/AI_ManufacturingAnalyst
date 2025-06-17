# --- START OF MODIFICATIONS ---

import streamlit as st
import pandas as pd
import time
from datetime import datetime
import html # For escaping debug output

# Import your own code
try:
    from mistral_app import (
        vn,
        extract_sql_from_response,
        generate_complex_sql,
        analyze_and_explain,
        current_date
    )
    if not hasattr(vn, 'generate_contextual_response'):
        st.error("Error: 'generate_contextual_response' missing in mistral_app.py")
        st.stop()
except ImportError as e:
    st.error(f"Import Error from mistral_app.py: {e}")
    st.stop()
except AttributeError as e:
    st.error(f"Attribute Error from mistral_app.py: {e}")
    st.stop()


# --- Page Config ---
st.set_page_config(page_title="Manufacturing SQL Agent", layout="wide")
st.title("🏭 FactoryTwin AI Agent")
st.subheader("Ask your manufacturing data questions")

# --- Initialize Session State Variables ---
default_values = {
    "question": "",
    "new_question": "",
    "category": "",
    "llm_response": "", # Still used, but maybe unreliable for suggestions across rerun
    "full_suggestion_content": "", # <--- NEW: Dedicated variable for suggestion display
    "extracted_sql": "",
    "df_result": None,
    "explanation": "",
    "current_step": "input_question",
    "execute_sql": False,
    "error_message": "",
    "plotly_figure": None
}
for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Helper Functions ---
def reset_app():
    """Resets all session state variables to their defaults."""
    for key, value in default_values.items():
        st.session_state[key] = value
    # No rerun here, let the caller handle rerun if needed

def set_step(step_name):
    """Sets the current step in the application flow."""
    st.session_state.current_step = step_name

# --- Main App Logic ---

# Step 1: Get user question
if st.session_state.current_step == "input_question":
    with st.form("question_form"):
        user_question = st.text_input(
            "Ask a manufacturing data question (type 'exit' to quit):",
            key="input_question_field",
            value=st.session_state.question # Preserve input
        )
        submit = st.form_submit_button("Submit Question")

        if submit:
            if user_question.lower().strip() == "exit":
                st.success("Session ended.")
                st.stop()
            elif not user_question.strip():
                st.warning("Please enter a question.")
            else:
                st.session_state.question = user_question
                st.session_state.new_question = f"Remember Today's date for date related queries {current_date}. Answer this question: {user_question}"
                # Clear potentially stale suggestion content before generating new
                st.session_state.full_suggestion_content = ""
                set_step("generate_sql")
                st.rerun()

# Step 2: Generate SQL and Initial Response
elif st.session_state.current_step == "generate_sql":
    st.info(f"Processing question: \"{st.session_state.question}\"")

    # Classify
    with st.spinner("Classifying question..."):
        try:
            st.session_state.category = vn.classify_question(st.session_state.new_question)
            st.write(f"**Classified Category:** `{st.session_state.category.upper()}`")
            st.warning(f"DEBUG (generate_sql): Classified as: {st.session_state.category}")
        except Exception as e:
            st.error(f"Classification Error: {e}")
            if st.button("Start Over"):
                reset_app()
                st.rerun()
            st.stop()

    # Generate response
    with st.spinner("Agent thinking... Generating SQL and analysis..."):
        try:
            if st.session_state.category == "suggestion":
                # Generate the full response (SQL + Text)
                full_response_text = vn.generate_contextual_response(st.session_state.new_question)
                st.session_state.llm_response = full_response_text # Store in original variable too
                st.session_state.full_suggestion_content = full_response_text # <--- Store in NEW dedicated variable
                st.warning("DEBUG (generate_sql): Stored full response in 'full_suggestion_content'") # <--- New DEBUG
            else: # Descriptive or Judgement
                st.session_state.llm_response = generate_complex_sql(st.session_state.new_question)
                st.session_state.full_suggestion_content = "" # Ensure suggestion variable is clear

            st.write("**LLM Response (Raw from Generation):**")
            st.text(st.session_state.llm_response) # DEBUG 2 (Should show full response for suggestion)
            response_len = len(st.session_state.llm_response) if st.session_state.llm_response else 0
            st.warning(f"DEBUG (generate_sql): Length of llm_response: {response_len}") # DEBUG 3

            if not st.session_state.llm_response:
                 st.error("LLM returned empty response.")
                 if st.button("Start Over"):
                     reset_app()
                     st.rerun()
                 st.stop()

        except Exception as e:
            st.error(f"LLM Generation Error: {e}")
            if st.button("Start Over"):
                reset_app()
                st.rerun()
            st.stop()

    # Extract SQL
    try:
        # Extract from the reliable full response if suggestion, otherwise from llm_response
        response_to_extract_from = st.session_state.full_suggestion_content if st.session_state.category == "suggestion" else st.session_state.llm_response
        extracted_sql = extract_sql_from_response(response_to_extract_from)
        st.session_state.extracted_sql = extracted_sql

        if not extracted_sql:
            st.error("Could not extract SQL from the LLM response.")
            st.info("LLM Response Content:")
            st.text(response_to_extract_from)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Try Asking Differently"):
                    set_step("input_question")
                    reset_app() # Reset to ensure clean state
                    st.rerun()
            with col2:
                if st.button("Cancel and Start Over"):
                    reset_app()
                    st.rerun()
        else:
            st.write("**Extracted SQL:**")
            st.code(extracted_sql, language="sql")

            # --- Check content before transition ---
            if st.session_state.category == "suggestion":
                st.warning("DEBUG (generate_sql): Preparing to move to handle_suggestion.")
                st.warning("Content in 'llm_response' before rerun:")
                st.text(st.session_state.llm_response) # DEBUG 5a
                st.warning("Content in 'full_suggestion_content' before rerun:")
                st.text(st.session_state.full_suggestion_content) # DEBUG 5b (Check the new variable)
                set_step("handle_suggestion")
            else:
                set_step("confirm_execution")
            st.rerun() # Trigger the transition

    except Exception as e:
        st.error(f"Error processing/extracting SQL: {str(e)}")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Try Asking Differently"):
                 set_step("input_question")
                 reset_app()
                 st.rerun()
        with col2:
            if st.button("Cancel and Start Over"):
                reset_app()
                st.rerun()

# Step 3a: Handle suggestion category (Display Only)
elif st.session_state.current_step == "handle_suggestion":
    # --- Check content upon entering step ---
    st.warning("DEBUG (handle_suggestion): Entered step.")
    st.warning("Content in 'llm_response' upon entry:")
    st.text(st.session_state.llm_response) # DEBUG 7a (Likely truncated)
    st.warning("Content in 'full_suggestion_content' upon entry:")
    st.text(st.session_state.get("full_suggestion_content", "NOT SET")) # DEBUG 7b (HOPEFULLY the full content)
    len_llm = len(st.session_state.llm_response) if st.session_state.llm_response else 0
    len_full = len(st.session_state.full_suggestion_content) if st.session_state.full_suggestion_content else 0
    st.warning(f"DEBUG (handle_suggestion): Length 'llm_response': {len_llm}") # DEBUG 8a
    st.warning(f"DEBUG (handle_suggestion): Length 'full_suggestion_content': {len_full}") # DEBUG 8b

    st.write("**Suggestion from Agent:**")
    # --- Use the NEW dedicated variable for display ---
    if st.session_state.get("full_suggestion_content"):
        st.text_area(
            "Suggestion and SQL Response",
            value=st.session_state.full_suggestion_content, # <--- Read from the new variable
            height=400,
            disabled=True,
            key="suggestion_display_area"
        )
    else:
        # Fallback or warning if the new variable is empty for some reason
        st.warning("Full suggestion content was not found in session state ('full_suggestion_content').")
        st.warning("Displaying potentially truncated content from 'llm_response':")
        st.text_area(
            "Suggestion and SQL Response (Fallback)",
            value=st.session_state.llm_response,
            height=200,
            disabled=True,
            key="suggestion_display_area_fallback"
        )


    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ I'm Satisfied"):
            reset_app()
            st.rerun()
    with col2:
        if st.button("🔄 Rethink Suggestion"):
            st.session_state.new_question = f"Please provide a different suggestion based on the previous context for the question: {st.session_state.question}"
            # Clear old suggestion content before regenerating
            st.session_state.full_suggestion_content = ""
            set_step("generate_sql")
            st.rerun()

# Step 3b: Confirm SQL execution
elif st.session_state.current_step == "confirm_execution":
    # ... (code remains largely the same, ensure it uses st.session_state.extracted_sql) ...
    if not st.session_state.extracted_sql:
        st.error("Cannot execute: No SQL query was extracted.")
        if st.button("Go Back"):
            set_step("generate_sql")
            st.rerun()
    else:
        st.write(f"**Ready to execute the following {st.session_state.category.upper()} query:**")
        st.code(st.session_state.extracted_sql, language="sql")
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"🚀 Execute SQL Query"):
                st.session_state.execute_sql = True
                set_step("execute_sql")
                st.rerun()
        with col2:
            if st.button("❌ Cancel Execution"):
                reset_app()
                st.rerun()


# Step 4: Execute SQL and show results
elif st.session_state.current_step == "execute_sql":
    # ... (code remains largely the same, ensure it uses st.session_state.extracted_sql) ...
    if not st.session_state.extracted_sql:
        st.error("Execution step reached without an SQL query.")
        if st.button("Start Over"):
            reset_app()
            st.rerun()
        st.stop()

    st.write("**Executing SQL query...**")
    st.code(st.session_state.extracted_sql, language="sql")

    try:
        with st.spinner("Running query..."):
            exec_start_time = time.time()
            df_result = vn.run_sql(sql=st.session_state.extracted_sql)
            st.session_state.df_result = df_result
            st.success(f"Query executed in {time.time() - exec_start_time:.2f} seconds. Found {len(df_result)} rows.")

        if df_result is None or df_result.empty:
            st.info("Query returned no results.")
        else:
            st.write("**Query Results:**")
            # ... (display dataframe) ...
            if len(df_result) > 20:
                st.dataframe(df_result.head(20))
                st.info(f"... (displaying first 20 of {len(df_result)} rows)")
            else:
                st.dataframe(df_result)

            # --- Visualization ---
            with st.spinner("Generating visualization..."):
                 try:
                     # Check if generate_plotly_code exists
                     if hasattr(vn, 'generate_plotly_code'):
                         plotly_code = vn.generate_plotly_code(
                             question=st.session_state.question,
                             sql=st.session_state.extracted_sql,
                             df=df_result
                         )
                         if plotly_code and hasattr(vn, 'get_plotly_figure'):
                             fig = vn.get_plotly_figure(plotly_code=plotly_code, df=df_result)
                             if fig:
                                 st.write("**Data Visualization:**")
                                 st.plotly_chart(fig, use_container_width=True)
                                 st.session_state.plotly_figure = fig
                             else:
                                 st.info("Could not create Plotly figure from generated code.")
                         elif plotly_code:
                              st.info("Plotly code generated, but 'get_plotly_figure' missing. Showing code:")
                              st.code(plotly_code, language='python')
                         else:
                             st.info("No visualization could be generated.")
                     else:
                          st.info("Visualization generation not available (vn.generate_plotly_code missing).")
                 except Exception as viz_error:
                     st.warning(f"Visualization error: {str(viz_error)}")

            # --- Analysis/Explanation ---
            if st.session_state.category in ["judgement", "descriptive"]:
                 with st.spinner("Analyzing results..."):
                    try:
                        explanation = analyze_and_explain(st.session_state.question, df_result, st.session_state.category)
                        st.session_state.explanation = explanation
                        st.write(f"**{st.session_state.category.upper()} ANALYSIS:**")
                        st.markdown(explanation)
                    except Exception as expl_error:
                        st.warning(f"Analysis generation error: {expl_error}")

        set_step("feedback")
        st.rerun()

    except Exception as e:
        st.error(f"Error executing SQL: {type(e).__name__} - {str(e)}")
        st.session_state.error_message = f"The query failed: ``````\nError: {type(e).__name__} - {str(e)}."
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Rethink Query (incorporating error)"):
                st.session_state.new_question = f"{st.session_state.error_message}\nPlease fix the SQL and address the original question: {st.session_state.question}"
                set_step("generate_sql")
                st.rerun()
        with col2:
            if st.button("🔁 Start Over"):
                reset_app()
                st.rerun()


# Step 5: Get feedback on results
elif st.session_state.current_step == "feedback":
    # ... (code remains largely the same) ...
    st.write("**Results and Explanation Review:**")
    # ... (display summary, plot, explanation) ...
    st.markdown("---")
    st.write("**Are you satisfied?**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Yes, Satisfied"):
            set_step("save_query")
            st.rerun()
    with col2:
        if st.button("🔄 No, Rethink Query"):
            st.session_state.new_question = f"Previous response unsatisfactory. Improve response for: {st.session_state.question}"
            st.session_state.df_result = None # Clear previous results
            st.session_state.plotly_figure = None
            st.session_state.explanation = ""
            set_step("generate_sql")
            st.rerun()


# Step 6: Save query option
elif st.session_state.current_step == "save_query":
    # ... (code remains largely the same) ...
    st.success("Analysis complete.")
    if st.session_state.extracted_sql:
        st.write("**Save this query?**")
        st.code(st.session_state.extracted_sql, language="sql")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Yes, Save"):
                 try:
                     if hasattr(vn, 'train') and callable(vn.train):
                         vn.train(question=st.session_state.question, sql=st.session_state.extracted_sql)
                         st.success("Query saved.")
                     else:
                          st.warning("Save functionality (vn.train) not available.")
                     time.sleep(1)
                     reset_app()
                     st.rerun()
                 except Exception as e:
                     st.error(f"Save failed: {e}")
                     if st.button("Continue without Saving"):
                          reset_app()
                          st.rerun()
        with col2:
            if st.button("💾 No, Don't Save"):
                reset_app()
                st.rerun()
    else:
        st.info("No SQL query generated.")
        if st.button("Start New Question"):
            reset_app()
            st.rerun()


# --- Persistent Debug Information ---
with st.expander("Show Debug Info", expanded=False):
    st.json({k: v for k, v in st.session_state.items() if k not in ['df_result', 'plotly_figure'] and not isinstance(v, pd.DataFrame)}) # Avoid large objects in json
    st.write("LLM Response (Raw Session State 'llm_response' - Escaped):")
    st.text(html.escape(st.session_state.get("llm_response", "Not Set")))
    st.write("Full Suggestion Content (Raw Session State 'full_suggestion_content' - Escaped):") # <--- Check the new variable here too
    st.text(html.escape(st.session_state.get("full_suggestion_content", "Not Set")))


# --- END OF MODIFICATIONS ---
