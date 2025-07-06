
import streamlit as st
import joblib
import pandas as pd

# Load model and features
model = joblib.load("rf_mental_health_model.pkl")
feature_list = joblib.load("rf_feature_list.pkl")

st.title("🧠 Mental Health Treatment Prediction App")

st.markdown("This app predicts whether an individual is likely to seek treatment for mental health based on workplace-related factors.")

# Input fields
age = st.slider("Age", 18, 70, 30)
gender = st.selectbox("Gender", ["Male", "Female", "Non-binary/Other"])
work_interfere = st.selectbox("How often does your mental health interfere with work?", ["Often", "Sometimes", "Rarely", "Never", "Unknown"])
remote_work = st.selectbox("Do you work remotely?", ["Yes", "No"])
leave = st.selectbox("Ease of taking mental health leave", ["Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult", "Don't know"])
benefits = st.selectbox("Does your employer provide mental health benefits?", ["Yes", "No", "Don't know"])
anonymity = st.selectbox("Is anonymity protected if you choose to seek mental health treatment?", ["Yes", "No", "Don't know"])

# Process input into a DataFrame
raw_input = {
    "Age": age,
    "Gender": gender,
    "work_interfere": work_interfere,
    "remote_work": remote_work,
    "leave": leave,
    "benefits": benefits,
    "anonymity": anonymity
}

df = pd.DataFrame([raw_input])

# Encode inputs
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0, "Non-binary/Other": 0})
df["remote_work"] = df["remote_work"].map({"Yes": 1, "No": 0})
df["benefits"] = df["benefits"].map({"Yes": 1, "No": 0, "Don't know": 0})
df["anonymity"] = df["anonymity"].map({"Yes": 1, "No": 0, "Don't know": 0})

# One-hot encode multi-categoricals
df = pd.get_dummies(df)

# Ensure all features are present
for col in feature_list:
    if col not in df.columns:
        df[col] = 0

df = df[feature_list]  # Align column order

# Predict
if st.button("Predict"):
    pred = model.predict(df)[0]
    if pred == 1:
        st.success("✅ Likely to Seek Mental Health Treatment")
    else:
        st.warning("❌ Not Likely to Seek Mental Health Treatment")
