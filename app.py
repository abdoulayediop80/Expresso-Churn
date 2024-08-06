# app.py
import streamlit as st
import joblib  # Use joblib for loading
import numpy as np

# Load the model
model_path = 'models/logistic_model.pkl'
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Prediction function
def predict_churn(region, tenure, montant, frequence_rech, revenue, arpu_segment, data_volume):
    input_data = np.array([[region, tenure, montant, frequence_rech, revenue, arpu_segment, data_volume]])
    return model.predict(input_data)

# Streamlit UI
st.title("Churn Prediction for Expresso")

region = st.selectbox("Region", [0, 1, 2, 3])
tenure = st.slider("Tenure", 0, 100, 10)
montant = st.number_input("Montant")
frequence_rech = st.number_input("Recharge Frequency")
revenue = st.number_input("Revenue")
arpu_segment = st.selectbox("ARPU Segment", [0, 1, 2])
data_volume = st.number_input("Data Volume")

if st.button("Predict"):
    prediction = predict_churn(region, tenure, montant, frequence_rech, revenue, arpu_segment, data_volume)
    st.write(f"Churn Prediction: {'Yes' if prediction[0] == 1 else 'No'}")