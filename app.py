import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load the saved regression model, classification model, scaler, and feature names
regression_model = joblib.load('lr.pkl')  # Regression model
scaler = joblib.load('sclr.pkl')  # Scaler
feature_names = joblib.load('feat.pkl')  # Feature names

# Streamlit app title
st.title("🏠 Housing Price Prediction App")
st.write("""
This app predicts the **housing price** and categorizes the house as **Affordable** or **Luxury** 
based on the predicted price.
""")

# Sidebar for user input
st.sidebar.header("Input Features")
st.sidebar.write("Please provide values for the following features:")

# Define the list of input features dynamically
input_features = {}

# Generate input fields for the features
for feature in feature_names:
    input_value = st.sidebar.number_input(f"{feature}:", value=0.0)
    input_features[feature] = input_value

# Convert inputs into a DataFrame for prediction
input_df = pd.DataFrame([input_features])

# Impute missing values (use mean for numeric features)
st.write("Imputing missing values...")
num_imputer = SimpleImputer(strategy='mean')
input_df_imputed = num_imputer.fit_transform(input_df)
input_df_imputed = pd.DataFrame(input_df_imputed, columns=feature_names)

# Scale the features using the saved scaler
st.write("Scaling features...")
input_df_scaled = scaler.transform(input_df_imputed)

# Define the price threshold for classification
price_threshold = 500000  # Adjust as needed

# Predict house price using the regression model
if st.button("Predict Price"):
    st.subheader("Prediction Results:")
    
    # Predict price
    predicted_price = regression_model.predict(input_df_scaled)[0]
    st.success(f"**Predicted Price**: ${predicted_price:,.2f}")
    
    # Predict category based on the price
    category = "Luxury" if predicted_price > price_threshold else "Affordable"
    st.subheader(f"**Category**: {category}")
    

# Footer
st.write("---")
st.write("Predict your house prices seamlessly!")
