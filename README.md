# 🧠 Mental Health Treatment Prediction — Full Pipeline + Live App

This project implements a complete end-to-end machine learning pipeline on the **Mental Health in Tech** dataset. The goal is to **predict whether an individual will seek mental health treatment** based on workplace, demographic, and organizational factors.

The final result is a **deployed web app** built using **Streamlit**, allowing users to input features and get real-time predictions.

## 📌 Table of Contents

- [Overview](#overview)
- [📊 Dataset Source](#dataset-source)
- [🔧 Task 1: Data Cleaning](#task-1-data-cleaning)
- [📊 Task 2: Exploratory Data Analysis (EDA)](#task-2-exploratory-data-analysis-eda)
- [🛠️ Task 3: Feature Engineering](#task-3-feature-engineering)
- [🤖 Task 4: Model Training](#task-4-model-training)
- [📈 Task 5: Feature Importance](#task-5-feature-importance)
- [🚀 Task 6: Deployment via FastAPI & Streamlit](#task-6-deployment-via-fastapi--streamlit)
- [🌐 Live Demo](#live-demo)
- [📁 Folder Structure](#folder-structure)
- [⚙️ How to Run Locally](#how-to-run-locally)

## 🧾 Overview

Mental health is often stigmatized, especially in the tech industry. Using real-world survey data, this project builds a robust machine learning model to **predict mental health treatment-seeking behavior** and make it available as a user-facing tool.

## 📊 Dataset Source

- Mental Health in Tech Survey (Kaggle / Open Source)

## 🔧 Task 1: Data Cleaning

- **Age**: Removed extreme outliers like `999999999`, and filtered values to the range `18–100`.
- **Gender**: Normalized over 30 messy categories into `Male`, `Female`, and `Non-binary/Other`, followed by one-hot encoding.
- **Self-employed**: Missing values filled based on supervisor status.
- **work_interfere**: Missing values filled with `"Unknown"`.
- **Irrelevant columns** like `comments`, `state`, `timestamp` dropped.

## 📊 Task 2: Exploratory Data Analysis (EDA)

Key insights from visualizations:

- **Gender vs Treatment**:
  - Females and non-binary individuals are more likely to seek help.
- **Workplace Interference**:
  - Higher frequency of interference correlates with higher treatment-seeking.
- **Leave Policy Awareness**:
  - “Don’t know” or “Very difficult” options correlate with increased treatment seeking.
- **Age Groups**:
  - Age was **not a strong predictor**, but skewed towards 25–35.

Visualizations included:
- Count plots, Box plots, Correlation heatmap, and 3D scatter plots colored by review scores, rating, and price.

## 🛠️ Task 3: Feature Engineering

- **Age Bracketing** into: `18–25`, `26–35`, `36–45`, `46–60`, `60+`
- **One-hot encoding** of:
  - `work_interfere`, `leave`, `benefits`, `care_options`, `anonymity`, `remote_work`, etc.
- **Target encoding**:
  - `treatment`: Yes → 1, No → 0
- Final model used **~40 features**.

## 🤖 Task 4: Model Training

Two models were implemented:

### 🔹 Logistic Regression
- **Accuracy**: 81.1%
- **Precision**: 75.5%
- **Recall**: 92.6%
- **F1 Score**: 83.2%

### 🔹 Random Forest (Deployed Model)
- **Accuracy**: 81.1%
- **Precision**: 77.1%
- **Recall**: 88.9%
- **F1 Score**: 82.6%

#### Model Selection:
- Logistic Regression had **better recall** → good for catching more at-risk individuals.
- Random Forest had **better precision** → used for deployment to reduce false positives.

## 📈 Task 5: Feature Importance (Random Forest)

Top 5 most influential features:
1. `work_interfere_Unknown`
2. `Age`
3. `work_interfere_Sometimes`
4. `care_options_Yes`
5. `remote_work_Yes`

## 🚀 Task 6: Deployment via FastAPI + Streamlit

This ML model was served through:
- ✅ `main.py` — FastAPI backend endpoint `/predict`
- ✅ `streamlit_app.py` — Beautiful interactive frontend for predictions
- ✅ `joblib` model: `rf_mental_health_model.pkl` and `rf_feature_list.pkl`

### 📦 Backend
- FastAPI used to load and predict from the model using structured JSON.

### 🎨 Frontend
- Built using Streamlit with:
  - Sliders for Age
  - Dropdowns for Gender, Leave policy, Anonymity, etc.
  - Real-time prediction output

## 🌐 Live Demo

> 💡 You can run this Streamlit app live (if deployed):
```
https://mental-health-app-dc5eewgzdhggwuqjda83ss.streamlit.app/
```

## 📁 Folder Structure

```
mental-health-app/
│
├── streamlit_app.py               # Frontend app
├── main.py                        # FastAPI backend (optional)
├── rf_mental_health_model.pkl     # Trained model
├── rf_feature_list.pkl            # Feature order for model
├── requirements.txt               # For deployment
├── README.md                      # This file
├── mental_health.ipynb            # Notebook with full pipeline
└── mental_health_cleaned.ipynb    # Cleaned version
```

## ⚙️ How to Run Locally

### 1. Clone this repo
```bash
git clone https://github.com/ProjectMLAI/mental-health-app.git
cd mental-health-app
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run streamlit_app.py
```

## 💬 Contact

> Project by **Abhishek Sinha**  
> Reach out via [LinkedIn](linkedin.com/in/abhishek-sinha-aa201829b) or raise issues for suggestions.
