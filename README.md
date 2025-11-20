🩺 Diabetes Prediction using Machine Learning

A simple ML project that predicts whether a person is diabetic based on medical features.
Built using Python, Pandas, Scikit-Learn, and Streamlit.


🚀 Project Features

Handles missing values (0 → NaN → Median Imputation)
Encodes categorical features (BMI category, Age group)
Scales numerical features using StandardScaler
Uses Logistic Regression for prediction
Interactive Streamlit Web App


📂 Dataset

768 rows × 9 columns
Target column: Outcome
0 → No Diabetes
1 → Diabetes
Missing values present as 0 in some columns: Glucose, BloodPressure, SkinThickness, Insulin, BMI


🧠 Model Workflow

Load dataset
Replace 0 with NaN
Apply median imputation
One-hot encode BMI & Age groups
Train-test split (80–20, stratified)
Scale features
Train Logistic Regression
Deploy with Streamlit


▶️ How to Run Locally

1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Run the Streamlit app
streamlit run app.py


🛠️ Tech Stack: Python, Pandas, NumPy, Scikit-Learn, Streamlit
