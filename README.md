# 🏨 Mental Health Treatment Prediction - Full Analysis

This document presents a complete data science pipeline applied to the **Mental Health in Tech Survey** dataset. The goal is to predict whether an individual will seek mental health treatment, based on workplace, demographic, and organizational support factors.

---

## 🔍 Task 1: Data Cleaning

1. **Age**: Outliers like `999999999` and values <18 and >100 were filtered out. After cleaning, the age distribution remained right-skewed, consistent with the tech workforce.
2. **Gender**: A normalization function was applied to convert over 30 variations of gender into three categories: `Male`, `Female`, and `Non-binary/Other`, followed by one-hot encoding.
3. **Missing Values**:

   * `self_employed` was imputed based on whether the person had a supervisor.
   * `work_interfere` was filled with `'Unknown'`.
   * Irrelevant columns like `comments`, `state`, and `timestamp` were dropped.

---

## 📊 Task 2: Exploratory Data Analysis (EDA)

Key insights extracted:

1. **Treatment Distribution**:

   * A large portion of the respondents reported seeking treatment.
   * Gender comparison showed females and non-binary individuals are more likely to seek help than males.

2. **Workplace Interference**:

   * Individuals whose mental health interferes with work (`Often` or `Sometimes`) are significantly more likely to seek treatment.
   * Those who answered `Never` or `Unknown` are the least likely.

3. **Leave Policies**:

   * People who find it difficult to take mental health leave are more likely to seek treatment externally.
   * Lack of clarity (`Don't know`) about leave options correlates with **lowest treatment seeking**.

4. **Age vs Treatment**:

   * No major difference in treatment behavior across age groups. Age is not a strong predictor.

---

## 🛠️ Task 3: Feature Engineering

1. **Age Bracketing**: Age was binned into 5 brackets: `18–25`, `26–35`, `36–45`, `46–60`, and `60+`.
2. **One-Hot Encoding**: Applied to all categorical columns including `self_employed`, `work_interfere`, `leave`, `benefits`, and others.
3. **Target Encoding**: `treatment` converted to binary (Yes → 1, No → 0).
4. **Final Feature Set**: Expanded to \~40 engineered variables.

---

## 🧬 Task 4: Modeling

### 🔹 Logistic Regression

* **Accuracy**: 81.1%
* **Precision**: 75.5%
* **Recall**: 92.6%
* **F1 Score**: **83.2%**

### 🔹 XGBoost

* **Accuracy**: 81.1%
* **Precision**: **77.1%**
* **Recall**: 88.9%
* **F1 Score**: 82.6%

**Interpretation**:

* Logistic Regression had higher recall and slightly better F1 score, making it more sensitive to identifying those who seek help.
* XGBoost had higher precision, meaning fewer false positives.

**Model Recommendation**:

* If the goal is to **catch at-risk individuals**, prioritize **recall** → Logistic Regression.
* If you want to reduce false alerts to HR or healthcare systems, use XGBoost.

---

## 📊 Task 5: (Next) Feature Importance & Explainability

* Will be used to interpret what variables are driving the predictions
* Will explore visualizations like:

  * Feature importance bar chart
  * SHAP values (optional)
  * Correlation heatmaps

---

## 🚀 Final Summary:

This project demonstrates a complete ML workflow:

* Dirty survey data was cleaned and structured using domain logic
* Meaningful insights were visualized through EDA
* Model performance was excellent, with high F1 and recall scores
* The pipeline is ready for model interpretation and even deployment

🚀 You're now interview-ready for product-grade ML evaluation!
