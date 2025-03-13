import streamlit as st
import pandas as pd

# Load modified student dataset
STUDENT_DATA_PATH = "student_performance_dataset_modified.xlsx"
df = pd.read_excel(STUDENT_DATA_PATH)

def check_credentials(register_number, password):
    register_number = register_number.strip().upper()  # Ensure case consistency
    student = df[(df['Register Number'] == register_number) & (df['Password'] == password)]
    return not student.empty

def get_student_scores(register_number):
    student = df[df['Register Number'] == register_number]
    if not student.empty:
        return {
            "MCQ Marks": student.iloc[0]['MCQ Score'],
            "Subjective Marks": student.iloc[0]['Subjective Score'],
            "Coding Marks": student.iloc[0]['Coding Score']
        }
    return None

# Streamlit Login Page
st.title("Student Login")
register_number = st.text_input("Register Number")
password = st.text_input("Password", type="password")

if st.button("Login"):
    if check_credentials(register_number, password):
        st.success("Login Successful! Redirecting...")

        # Store login details in session state
        st.session_state["logged_in"] = True
        st.session_state["register_number"] = register_number

        # Redirect to grad.py
        st.experimental_set_query_params(logged_in=True, reg_number=register_number)
        st.switch_page("grad3.py")  # Only works if using Streamlit multipage
    else:
        st.error("Invalid Register Number or Password")
