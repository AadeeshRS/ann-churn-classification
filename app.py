import os
import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Attempt a compatible model load
try:
    from tensorflow.keras.models import load_model
    _load_model = lambda p: load_model(p)
except Exception:
    import keras
    _load_model = lambda p: keras.models.load_model(p)

st.set_page_config(page_title="Customer Churn Predictor", layout="wide", initial_sidebar_state="expanded")

# Small CSS to tidy layout
st.markdown(
    """
    <style>
    .stApp { font-family: "Segoe UI", Roboto, Arial; }
    .big-title { font-size:34px; font-weight:600; margin-bottom:6px; }
    .subtle { color: #6c757d; margin-bottom:18px; }
    .metric-label { font-weight:600; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_resources():
    model = _load_model('model.h5')
    with open('ohe_geo.pkl', 'rb') as f:
        ohe_geo = pickle.load(f)
    with open('label_encoder_gender.pkl', 'rb') as f:
        label_encoder_gender = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, ohe_geo, label_encoder_gender, scaler

model, ohe_geo, label_encoder_gender, scaler = load_resources()

# Sidebar: user inputs
with st.sidebar.form(key="input_form"):
    st.header("Customer Input")
    geography = st.selectbox('Geography', ohe_geo.categories_[0])
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    age = st.slider('Age', 18, 92, 35)
    credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=650)
    balance = st.number_input('Balance', min_value=0.0, value=50000.0, format="%.2f")
    estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=70000.0, format="%.2f")
    tenure = st.slider('Tenure (years)', 0, 10, 3)
    num_of_products = st.slider('Number of Products', 1, 4, 1)
    has_cr_card = st.selectbox('Has Credit Card', [0, 1], index=1)
    is_active_member = st.selectbox('Is Active Member', [0, 1], index=1)
    submitted = st.form_submit_button("Predict")

# Main layout
st.markdown('<div class="big-title">Customer Churn Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Enter customer details in the sidebar and click Predict.</div>', unsafe_allow_html=True)

if submitted:
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    geo_encoded = ohe_geo.transform([[geography]])
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe_geo.get_feature_names_out(['Geography']))
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)
    prediction_proba = float(prediction[0][0])
    prob_percent = prediction_proba * 100

    # Top row: metric + progress
    col1, col2 = st.columns([2, 3])
    with col1:
        st.metric(label="Churn Probability", value=f"{prob_percent:.1f}%")
        status = "At Risk" if prediction_proba > 0.5 else "Low Risk"
        if prediction_proba > 0.5:
            st.error(f"Status: {status}")
        else:
            st.success(f"Status: {status}")

    with col2:
        st.write("Probability visual")
        st.progress(min(int(prob_percent), 100))

    # Detailed info and raw output
    with st.expander("View input data"):
        st.dataframe(input_data.style.format("{:.3f}"))

    with st.expander("Prediction details"):
        st.write({"raw_probability": prediction_proba})
        st.write("Decision threshold: 0.5")

    # Optional improvement: simple suggestion
    if prediction_proba > 0.5:
        st.info("Suggested actions: offer retention incentives, contact support team, flag account for review.")
else:
    st.info("Fill the form in the sidebar and press Predict to get a prediction.")