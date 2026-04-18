"""
Streamlit Application for Telco Customer Churn Prediction
Interactive UI to test the churn prediction model directly
"""

import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📱",
    layout="wide"
)

# Load model and preprocessing artifacts
@st.cache_resource
def load_model():
    """Load the trained model and preprocessing artifacts."""
    try:
        # Load the best model
        model = joblib.load('best_model.pkl')
        
        # Load preprocessing artifacts
        scaler = joblib.load('scaler.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        
        return model, scaler, feature_columns
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

model, scaler, feature_columns = load_model()

# Title
st.title("📱 Telco Customer Churn Predictor")
st.markdown("Predict whether a customer will churn based on their profile")

# Sidebar with information
st.sidebar.header("About the app")
st.sidebar.markdown("""
This app uses a machine learning model to predict customer churn for a telco company.

**Model**: Random Forest (or your best model)  
**Features**: Customer demographics, service usage, billing info  
**Deployment**: Streamlit Cloud
""")

if model is None:
    st.stop()

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
    
    # Convert to DataFrame and preprocess
    df = pd.DataFrame([input_data])
    
    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # Ensure all expected columns are present
    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Reorder columns to match training
    df_encoded = df_encoded[feature_columns]
    
    # Scale the features
    df_scaled = scaler.transform(df_encoded)
    
    # Make prediction
    prediction_proba = model.predict_proba(df_scaled)[0]
    churn_probability = prediction_proba[1]
    prediction = "Churn" if churn_probability > 0.5 else "No Churn"
    
    # Display results
    st.success("Prediction completed!")
    
    col_result1, col_result2 = st.columns(2)
    
    with col_result1:
        st.metric("Prediction", prediction)
        
    with col_result2:
        st.metric("Churn Probability", f"{churn_probability:.1%}")
    
    # Progress bar for probability
    st.progress(float(churn_probability))
    
    # Additional info
    if prediction == "Churn":
        st.warning("⚠️ This customer is likely to churn. Consider retention strategies!")
    else:
        st.success("✅ This customer is likely to stay. Great!")

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit and scikit-learn*")