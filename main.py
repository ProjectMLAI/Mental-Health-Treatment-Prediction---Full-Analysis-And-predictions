
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


# Load model and features
model = joblib.load("rf_mental_health_model.pkl")
features = joblib.load("rf_feature_list.pkl")


# Define expected raw inputs from user
class InputData(BaseModel):
    Age: int
    Gender: str
    work_interfere: str
    remote_work: str
    leave: str
    benefits: str
    anonymity: str
    # Add more fields if needed based on what your model uses

# Initialize app
app = FastAPI()

# Define raw → processed mapping
def preprocess_input(data: dict):
    df = pd.DataFrame([data])

    # Manual encoding as per your training pipeline
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0, 'Non-binary/Other': 0}).fillna(0)
    df['remote_work'] = df['remote_work'].map({'Yes': 1, 'No': 0}).fillna(0)
    df['benefits'] = df['benefits'].map({'Yes': 1, 'No': 0, "Don't know": 0}).fillna(0)
    df['anonymity'] = df['anonymity'].map({'Yes': 1, 'No': 0, "Don't know": 0}).fillna(0)

    # Use get_dummies for multi-categoricals like work_interfere, leave
    df = pd.get_dummies(df)

    # Ensure all columns are in correct order and fill missing ones
    for col in features:
        if col not in df.columns:
            df[col] = 0
    df = df[features]

    return df

# Prediction endpoint
@app.post("/predict")
def predict(input_data: InputData):
    data_dict = input_data.dict()
    processed_df = preprocess_input(data_dict)
    prediction = model.predict(processed_df)[0]
    return {"prediction": int(prediction)}
