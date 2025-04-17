import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('house_model.pkl', 'rb') as file:
    model = pickle.load(file)

# ---------- Custom CSS Styling ----------
st.markdown("""
    <style>
        /* Background */
        .stApp {
            background: linear-gradient(to right, #e0eafc, #cfdef3);
            color: #000000;
            font-family: 'Segoe UI', sans-serif;
        }

        /* Title Styling */
        .title-wrapper {
            background-color: #4a90e2;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            color: white;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        }

        /* Input Label */
        label {
            font-size: 18px;
            color: #333333;
        }

        /* Buttons */
        .stButton > button {
            background-color: #4a90e2;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 8px;
        }

        .stButton > button:hover {
            background-color: #3a70b2;
        }

        /* Success Message */
        .stAlert-success {
            background-color: #d0f0c0;
            color: #1e4620;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Streamlit App Content ----------
st.markdown('<div class="title-wrapper"><h1>🏠 House Price Prediction App</h1></div>', unsafe_allow_html=True)

st.write("#### 📊 This app predicts the house price based on square footage using a Simple Linear Regression model.")

# Input for square footage
sqft_living = st.number_input(
    "Enter Square Foot Living Area (sqft):",
    min_value=100.0, max_value=10000.0,
    value=1000.0, step=50.0
)

# Predict button
if st.button("Predict Price"):
    sqft_input = np.array([[sqft_living]])
    prediction = model.predict(sqft_input)

    st.success(f"💰 Predicted house price for **{sqft_living:.0f} sqft** is: **${prediction[0]:,.2f}**")

# Additional info
st.markdown("---")
st.markdown("ℹ️ *This model was trained using historical house data with Linear Regression.*")
