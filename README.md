# Telco Customer Churn Prediction

This project demonstrates a complete machine learning workflow for predicting customer churn using the Telco Customer Churn dataset. It includes MLflow experiment tracking, FastAPI deployment, and a Streamlit web interface.

## Project Structure

- `run2.py` - Main training script with MLflow experiment tracking
- `app.py` - FastAPI application for model serving
- `streamlit_app.py` - Streamlit web interface for model testing
- `requirements.txt` - Python dependencies
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` - Dataset
- `scaler.pkl` - Preprocessing scaler
- `feature_columns.pkl` - Feature column names
- `mlruns/` - MLflow experiment data

## Features

- **Data Preprocessing**: Imputation, one-hot encoding, feature scaling
- **Model Training**: Logistic Regression, Decision Tree, Random Forest
- **Experiment Tracking**: MLflow for logging parameters, metrics, and artifacts
- **Model Deployment**: FastAPI REST API
- **Web Interface**: Streamlit app for interactive predictions

## Setup and Installation

1. **Clone/Create the project** and navigate to the directory

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the models** (if not already done):
   ```bash
   python run2.py
   ```

## Running the Applications

### 1. Start the FastAPI Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: http://localhost:8000

**API Endpoints:**
- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /predict` - Make predictions

**API Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 2. Start the Streamlit App

```bash
streamlit run streamlit_app.py
```

The web interface will be available at: http://localhost:8501

## Usage

### Via Streamlit Web Interface

1. Open http://localhost:8501 in your browser
2. Fill in the customer information in the form
3. Click "Predict Churn"
4. View the prediction results and churn probability

### Via FastAPI (Programmatic)

```python
import requests

# Example customer data
customer_data = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 24,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 2045.0
}

response = requests.post("http://localhost:8000/predict", json=customer_data)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Churn Probability: {result['churn_probability']:.1%}")
```

## Model Performance

Based on the latest training run:

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.8204 | 0.6852 | 0.5952 | 0.6370 |
| Decision Tree | 0.7935 | 0.6323 | 0.5255 | 0.5739 |
| Random Forest | 0.8084 | 0.6873 | 0.5067 | 0.5833 |

**Best Model**: Logistic Regression (F1: 0.6370)

## MLflow Tracking

View experiment results in MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open: http://localhost:5000

## Dataset

The project uses the Telco Customer Churn dataset from IBM, which contains information about telecom customers and whether they churned.

**Features:**
- Customer demographics (gender, senior citizen, partner, dependents)
- Service information (phone, internet, security, support, etc.)
- Billing information (tenure, monthly charges, total charges)
- Contract and payment details

## Technologies Used

- **Python** - Core language
- **Pandas & NumPy** - Data processing
- **Scikit-learn** - Machine learning
- **MLflow** - Experiment tracking
- **FastAPI** - REST API framework
- **Streamlit** - Web interface
- **Uvicorn** - ASGI server

## License

This project is for educational purposes as part of Machine Learning Design for Business course.