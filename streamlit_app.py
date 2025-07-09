import streamlit as st
import joblib
import pandas as pd
import numpy as np
from fpdf import FPDF
from datetime import datetime
import base64

# Load model and features
model = joblib.load("logreg_mental_health_model.pkl")
feature_list = joblib.load("logreg_feature_list.pkl")

st.title(" Mental Health Risk Profiler")

st.markdown("Predict whether someone is likely to seek mental health treatment — and understand why.")

# --- Input Form ---
age = st.slider("Age", 18, 70, 30)
gender = st.selectbox("Gender", ["Male", "Female", "Non-binary/Other"])
work_interfere = st.selectbox("How often does your mental health interfere with work?", ["Often", "Sometimes", "Rarely", "Unknown"])
remote_work = st.selectbox("Do you work remotely?", ["Yes", "No"])
leave = st.selectbox("Ease of taking mental health leave", ["Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult"])
benefits = st.selectbox("Does your employer provide mental health benefits?", ["Yes", "No", "Don't know"])
care_options = st.selectbox("Does your employer provide care options for mental health?", ["Yes", "No", "Not sure"])
wellness_program = st.selectbox("Is there a mental health wellness program?", ["Yes", "No", "Don't know"])
anonymity = st.selectbox("Is anonymity protected when seeking treatment?", ["Yes", "No", "Don't know"])
mental_health_consequence = st.selectbox("Would seeking treatment negatively impact your job?", ["Yes", "No"])
family_history = st.selectbox("Do you have a family history of mental illness?", ["Yes", "No"])
self_employed = st.selectbox("Are you self-employed?", ["Yes", "No"])

# --- Input Processing ---
df = pd.DataFrame([{
    "log_age": np.log(age + 1),
    "gender_Male": 1 if gender == "Male" else 0,
    "gender_Non-binary/Other": 1 if gender == "Non-binary/Other" else 0,
    "self_employed_Yes": 1 if self_employed == "Yes" else 0,
    "remote_work_Yes": 1 if remote_work == "Yes" else 0,
    "work_interfere_Often": 1 if work_interfere == "Often" else 0,
    "work_interfere_Sometimes": 1 if work_interfere == "Sometimes" else 0,
    "work_interfere_Rarely": 1 if work_interfere == "Rarely" else 0,
    "work_interfere_Unknown": 1 if work_interfere == "Unknown" else 0,
    "leave_Very easy": 1 if leave == "Very easy" else 0,
    "leave_Somewhat easy": 1 if leave == "Somewhat easy" else 0,
    "leave_Somewhat difficult": 1 if leave == "Somewhat difficult" else 0,
    "leave_Very difficult": 1 if leave == "Very difficult" else 0,
    "benefits_Yes": 1 if benefits == "Yes" else 0,
    "benefits_No": 1 if benefits == "No" else 0,
    "care_options_Yes": 1 if care_options == "Yes" else 0,
    "care_options_Not sure": 1 if care_options == "Not sure" else 0,
    "wellness_program_Yes": 1 if wellness_program == "Yes" else 0,
    "wellness_program_No": 1 if wellness_program == "No" else 0,
    "anonymity_Yes": 1 if anonymity == "Yes" else 0,
    "anonymity_No": 1 if anonymity == "No" else 0,
    "mental_health_consequence_Yes": 1 if mental_health_consequence == "Yes" else 0,
    "mental_health_consequence_No": 1 if mental_health_consequence == "No" else 0,
    "family_history": 1 if family_history == "Yes" else 0,
    "support_score": (
        2 * (1 if benefits == "Yes" else 0) +
        2 * (1 if care_options == "Yes" else 0) +
        1 * (1 if care_options == "Not sure" else 0) +
        2 * (1 if wellness_program == "Yes" else 0)
    ),
    "high_risk_workplace": (
        1 if work_interfere == "Often" and mental_health_consequence == "Yes" else 0
    )
}])

# --- Align Columns ---
for col in feature_list:
    if col not in df.columns:
        df[col] = 0
df = df[feature_list]

# --- PDF Generator ---
def generate_pdf_report(input_data, prediction, probability, recommendation):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "Mental Health Risk Report", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, "User Inputs:", ln=True)
    pdf.set_font("Arial", size=12)
    for key, value in input_data.items():
        pdf.cell(200, 10, f"{key}: {value}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, "Prediction Result:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"{prediction} ({probability:.2f}% likelihood)", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, "Recommendation:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, recommendation)

    pdf.output("report.pdf")
    with open("report.pdf", "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    return base64_pdf

# --- Predict ---
if st.button("Predict"):
    proba = model.predict_proba(df)[0][1] * 100
    label = model.predict(df)[0]

    st.subheader("🧠 Prediction Result")
    if label == 1:
        st.success(f"Likely to Seek Treatment ({proba:.2f}%)")
    else:
        st.warning(f"Not Likely to Seek Treatment ({proba:.2f}%)")

    # Smart Recommendation
    if proba >= 75:
        recommendation = "You are at high risk. Please consult your HR or a mental health counselor proactively."
    elif proba >= 50:
        recommendation = "You may benefit from increased mental health support and self-monitoring."
    else:
        recommendation = "Your environment seems stable. Stay self-aware and informed."

    st.info(recommendation)

    # PDF Summary
    input_summary = {
        "Age": age,
        "Gender": gender,
        "Family History": family_history,
        "Work Interfere": work_interfere,
        "Remote Work": remote_work,
        "Leave Policy": leave,
        "Benefits": benefits,
        "Care Options": care_options,
        "Wellness Program": wellness_program,
        "Anonymity": anonymity,
        "Job Consequence": mental_health_consequence,
        "Self-employed": self_employed
    }

    pdf_data = generate_pdf_report(
        input_data=input_summary,
        prediction="Likely to Seek Treatment" if label == 1 else "Not Likely to Seek Treatment",
        probability=proba,
        recommendation=recommendation
    )

    st.download_button(
        label="📄 Download PDF Report",
        data=base64.b64decode(pdf_data),
        file_name="mental_health_risk_report.pdf",
        mime="application/pdf"
    )
