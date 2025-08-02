import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="auto",
)


try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Error: 'model.pkl' not found. Please ensure the model file is in the same directory.")
    st.stop()


st.title("Medical Insurance Cost Predictor 🏥")

st.write(
    "Enter the patient's details below to get an estimated insurance cost. "
    #"This app uses a Random Forest Regressor model to make predictions."
)

# --- User Inputs---
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        bmi = st.number_input("BMI (Body Mass Index)", min_value=15.0, max_value=55.0, value=25.0, step=0.1)
        sex = st.selectbox("Sex", ("male", "female"))

    with col2:
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, step=1)
        smoker = st.selectbox("Smoker", ("yes", "no"))
        region = st.selectbox("Region", ("southwest", "southeast", "northwest", "northeast"))

    # Submit button for the form
    submit_button = st.form_submit_button(label="Predict Cost")

if submit_button:
    # --- Data Preprocessing for Prediction ---
    # This must match the preprocessing in your training script

    # 1. Encode binary features
    sex_encoded = 1 if sex == "male" else 0
    smoker_encoded = 1 if smoker == "yes" else 0

    # 2. One-Hot Encode the region
    # The model was trained with 'northeast' as the dropped category
    region_northwest = 1 if region == "northwest" else 0
    region_southeast = 1 if region == "southeast" else 0
    region_southwest = 1 if region == "southwest" else 0

    # 3. Create the input DataFrame for the model
    # The column order must be exactly the same as in the training data
    input_data = pd.DataFrame(
        [
            [
                age,
                sex_encoded,
                bmi,
                children,
                smoker_encoded,
                region_northwest,
                region_southeast,
                region_southwest,
            ]
        ],
        columns=[
            "age", "sex", "bmi", "children", "smoker",
            "region_northwest", "region_southeast", "region_southwest"
        ],
    )

    # --- Prediction and Display ---
    prediction_usd = model.predict(input_data)[0]
    usd_to_inr_rate = 83
    prediction_inr = prediction_usd * usd_to_inr_rate

    st.success(f"**Predicted Insurance Cost: ₹{prediction_inr:,.2f}**")
