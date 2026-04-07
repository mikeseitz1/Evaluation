"""
Streamlit Application for Telco Customer Churn Prediction
Interactive UI to test the churn prediction model via FastAPI
"""

import streamlit as st
import requests
import json

# Page configuration
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📱",
    layout="wide"
)

# Title
st.title("📱 Telco Customer Churn Predictor")
st.markdown("Predict whether a customer will churn based on their profile")

# Sidebar with information
st.sidebar.header("About the app")
st.sidebar.markdown("""
This app uses a machine learning model to predict customer churn for a telco company.

**Model**: Logistic Regression  
**Features**: Customer demographics, service usage, billing info  
**API**: FastAPI backend
""")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Customer Demographics")
    
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    
    st.subheader("📞 Services")
    
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

with col2:
    st.subheader("🛡️ Security & Support")
    
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    
    st.subheader("📺 Entertainment")
    
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

# Billing information
st.subheader("💳 Billing Information")

col3, col4, col5 = st.columns(3)

with col3:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)

with col4:
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0, step=0.01)

with col5:
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=600.0, step=0.01)

# Contract and payment
col6, col7 = st.columns(2)

with col6:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

with col7:
    payment_method = st.selectbox("Payment Method", 
                                 ["Electronic check", "Mailed check", 
                                  "Bank transfer (automatic)", "Credit card (automatic)"])

# Prediction button
if st.button("🔮 Predict Churn", type="primary"):
    # Prepare input data
    input_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }
    
    # Make API request
    try:
        response = requests.post("http://localhost:8000/predict", json=input_data)
        
        if response.status_code == 200:
            result = response.json()
            
            # Display results
            st.success("Prediction completed!")
            
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                st.metric("Prediction", result["prediction"])
                
            with col_result2:
                st.metric("Churn Probability", f"{result['churn_probability']:.1%}")
            
            # Progress bar for probability
            st.progress(result["churn_probability"])
            
            # Additional info
            if result["prediction"] == "Churn":
                st.warning("⚠️ This customer is likely to churn. Consider retention strategies!")
            else:
                st.success("✅ This customer is likely to stay. Great!")
                
        else:
            st.error(f"API Error: {response.status_code}")
            st.text(response.text)
            
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {e}")
        st.info("Make sure the FastAPI server is running on http://localhost:8000")

# Footer
st.markdown("---")
st.markdown("*Built with FastAPI, Streamlit, and MLflow*")