"""
FastAPI Application for Telco Customer Churn Prediction
Includes comprehensive input validation and anomaly detection
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from typing import Literal
from validation import CrossFieldValidator, OutlierDetector

# Initialize FastAPI app
app = FastAPI(title="Telco Churn Prediction API", version="1.0.0")

# Load preprocessing artifacts
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# Load the best model (Logistic Regression)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
experiment = mlflow.get_experiment_by_name("Telco Customer Churn Prediction")
if experiment:
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    # Find Logistic Regression run
    lr_run = runs[runs['tags.mlflow.runName'] == 'Logistic Regression'].iloc[0]
    run_id = lr_run['run_id']
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
else:
    raise ValueError("Experiment not found")

# Define input data model
class ChurnPredictionInput(BaseModel):
    gender: Literal['Male', 'Female']
    SeniorCitizen: int
    Partner: Literal['Yes', 'No']
    Dependents: Literal['Yes', 'No']
    tenure: int
    PhoneService: Literal['Yes', 'No']
    MultipleLines: Literal['Yes', 'No', 'No phone service']
    InternetService: Literal['DSL', 'Fiber optic', 'No']
    OnlineSecurity: Literal['Yes', 'No', 'No internet service']
    OnlineBackup: Literal['Yes', 'No', 'No internet service']
    DeviceProtection: Literal['Yes', 'No', 'No internet service']
    TechSupport: Literal['Yes', 'No', 'No internet service']
    StreamingTV: Literal['Yes', 'No', 'No internet service']
    StreamingMovies: Literal['Yes', 'No', 'No internet service']
    Contract: Literal['Month-to-month', 'One year', 'Two year']
    PaperlessBilling: Literal['Yes', 'No']
    PaymentMethod: Literal['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
    MonthlyCharges: float
    TotalCharges: float

def preprocess_input(data: ChurnPredictionInput) -> tuple:
    """
    Preprocess input data to match training format with validation.
    
    Returns:
        Tuple of (processed_data, validation_warnings)
    """
    warnings = []
    
    # Convert to DataFrame for validation
    df = pd.DataFrame([data.dict()])
    
    # Cross-field validation: tenure vs charges
    expected_total = data.MonthlyCharges * data.tenure
    if data.tenure > 0:
        relative_diff = abs(data.TotalCharges - expected_total) / expected_total
        if relative_diff > 0.1:  # 10% tolerance
            warnings.append(
                f"TotalCharges ({data.TotalCharges:.2f}) differs from expected "
                f"({expected_total:.2f}) by {relative_diff*100:.1f}%"
            )
    
    # Service consistency validation
    if data.InternetService == 'No':
        internet_services = [
            data.OnlineSecurity, data.OnlineBackup, data.DeviceProtection,
            data.TechSupport, data.StreamingTV, data.StreamingMovies
        ]
        if any(s != 'No internet service' for s in internet_services):
            warnings.append(
                "InternetService='No' but some internet services are enabled. "
                "This is inconsistent and may affect predictions."
            )
    
    if data.PhoneService == 'No' and data.MultipleLines != 'No phone service':
        warnings.append(
            "PhoneService='No' but MultipleLines is not 'No phone service'. "
            "This is inconsistent."
        )
    
    # Check for outliers in numeric fields
    outlier_detector = OutlierDetector()
    numeric_issues = []
    
    if outlier_detector.detect_outliers_iqr(pd.Series([data.tenure]))[0]:
        numeric_issues.append("tenure")
    if outlier_detector.detect_outliers_iqr(pd.Series([data.MonthlyCharges]))[0]:
        numeric_issues.append("MonthlyCharges")
    if outlier_detector.detect_outliers_iqr(pd.Series([data.TotalCharges]))[0]:
        numeric_issues.append("TotalCharges")
    
    if numeric_issues:
        warnings.append(
            f"Potential anomalies detected in: {', '.join(numeric_issues)}. "
            "Prediction may be less reliable for unusual input values."
        )
    
    # Convert to DataFrame
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # Ensure all expected columns are present
    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Reorder columns to match training
    df_encoded = df_encoded[feature_columns]
    
    # Scale
    df_scaled = scaler.transform(df_encoded)
    
    return df_scaled, warnings

@app.post("/predict")
def predict_churn(data: ChurnPredictionInput):
    """Predict customer churn probability with validation warnings."""
    try:
        # Preprocess input with validation
        processed_data, validation_warnings = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]  # Probability of churn (class 1)
        
        return {
            "prediction": "Churn" if prediction == 1 else "No Churn",
            "churn_probability": float(probability),
            "confidence": float(max(probability, 1-probability)),
            "validation_warnings": validation_warnings,
            "warning_count": len(validation_warnings)
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def root():
    """Root endpoint."""
    return {"message": "Telco Churn Prediction API", "status": "running"}

@app.post("/validate")
def validate_input(data: ChurnPredictionInput):
    """Validate input data without making predictions."""
    try:
        # Preprocess input with validation
        _, validation_warnings = preprocess_input(data)
        
        return {
            "valid": len(validation_warnings) == 0,
            "validation_warnings": validation_warnings,
            "warning_count": len(validation_warnings),
            "message": "Input is clean" if len(validation_warnings) == 0 else f"Found {len(validation_warnings)} validation issue(s)"
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health():
    """Health check endpoint with model info."""
    return {
        "status": "healthy",
        "model": "Logistic Regression",
        "features": len(feature_columns),
        "validation": "Enabled",
        "checks": ["schema", "cross_field", "outliers"]
    }