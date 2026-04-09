# Telco Customer Churn Prediction

This project demonstrates a complete machine learning workflow for predicting customer churn using the Telco Customer Churn dataset. It includes MLflow experiment tracking, FastAPI deployment, and a Streamlit web interface.

## Project Structure

- `run2.py` - Main training script with MLflow experiment tracking
- `app.py` - FastAPI application for model serving
- `streamlit_app.py` - Streamlit web interface for model testing
- `requirements.txt` - Python dependencies
- `dvc.yaml` - DVC pipeline configuration
- `params.yaml` - Centralized hyperparameter and configuration file
- **`validation.py`** - Comprehensive validation utilities (schema, outlier detection, cross-field validation)
- **`startup.bat`** - Quick start script for Windows (batch)
- **`startup.ps1`** - Quick start script for Windows (PowerShell)
- **`startup.py`** - Quick start script for all platforms (Python)
- **`startup.sh`** - Quick start script for Linux/Mac (bash)
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` - Dataset
- `scaler.pkl` - Preprocessing scaler (tracked with DVC)
- `feature_columns.pkl` - Feature column names (tracked with DVC)
- `mlruns/` - MLflow experiment data
- `.dvc/` - DVC configuration and cache
- `DVC_SETUP.md` - Detailed DVC usage guide
- **`VALIDATION.md`** - Complete validation documentation

## Features

- **Data Preprocessing**: Imputation, one-hot encoding, feature scaling
- **Model Training**: Logistic Regression, Decision Tree, Random Forest
- **Experiment Tracking**: MLflow for logging parameters, metrics, and artifacts
- **Data Versioning**: DVC for tracking datasets and model artifacts
- **Reproducible Pipelines**: DVC pipeline definitions with params.yaml for consistent experiments
- **Model Deployment**: FastAPI REST API
- **Web Interface**: Streamlit app for interactive predictions

## Setup and Installation

1. **Clone/Create the project** and navigate to the directory

2. **Install dependencies** (including DVC):
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize DVC** (already configured, but to sync data):
   ```bash
   dvc pull  # Download tracked data files from remote storage
   ```

4. **Train the models** (if not already done):
   ```bash
   python run2.py
   ```

   Or use DVC to run the reproducible pipeline:
   ```bash
   dvc repro
   ```

## Running the Applications

### Quick Start - One Command

Start all services with a single command:

**Windows (Batch file):**
```bash
startup.bat
```

**Windows (PowerShell):**
```powershell
.\startup.ps1
```

**Windows (Python - any OS):**
```bash
python startup.py
```

**Linux/Mac:**
```bash
bash startup.sh
```

This will automatically start:
- FastAPI Server (port 8000)
- Streamlit App (port 8501)
- MLflow UI (port 5000) - interactive prompt on Windows batch/PowerShell

### Individual Service Startup

If you prefer to start services individually:

#### 1. Start the FastAPI Server

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

#### 2. Start the Streamlit App

```bash
streamlit run streamlit_app.py
```

The web interface will be available at: http://localhost:8501

## DVC Pipeline & Data Versioning

### Reproducible Pipeline

The project uses DVC for data versioning and pipeline reproducibility. All pipeline stages and hyperparameters are defined in configuration files:

- **`dvc.yaml`** - Pipeline stages and dependencies
- **`params.yaml`** - Centralized hyperparameters and configuration

### DVC Commands

```bash
# View the entire pipeline structure
dvc dag

# Run the complete reproducible pipeline
dvc repro

# Push data to remote storage (requires remote configuration)
dvc push

# Pull data from remote storage
dvc pull

# Add new files to DVC tracking
dvc add path/to/file

# View data versioning status
dvc status
```

### Run with Different Parameters

Modify `params.yaml` to adjust hyperparameters, then run:

```bash
dvc repro
```

Or override parameters directly:
```bash
dvc repro --set-param train.test_size=0.25
```

### Remote Storage Configuration

To enable collaboration and backup, configure a remote:

```bash
# AWS S3 example
dvc remote add -d myremote s3://my-bucket/dvc-storage

# Google Drive example
dvc remote add -d myremote gdrive://folder-id

# Push/pull with remote
dvc push
dvc pull
```

For detailed DVC setup and configuration, see **[DVC_SETUP.md](DVC_SETUP.md)**.

## Data & Model Validation

### Comprehensive Validation System

The project includes multi-level validation to ensure data quality and model reliability:

**1. Training Data Validation**
- Schema validation (required columns, data types)
- Outlier detection using Interquartile Range (IQR)
- Cross-field validation (tenure vs charges consistency)
- Service consistency checks

**2. API Input Validation**
- Type validation via Pydantic models
- Cross-field relationship checks
- Anomaly detection with warnings
- New `/validate` endpoint for pre-prediction checks

**3. Model Performance Validation**
- Models only logged if they meet minimum thresholds:
  - Accuracy ≥ 75%
  - Precision ≥ 60%
  - Recall ≥ 50%
  - F1 Score ≥ 55%

### API Prediction with Validation

The `/predict` endpoint now returns validation warnings:

```json
{
  "prediction": "Churn",
  "churn_probability": 0.72,
  "confidence": 0.72,
  "validation_warnings": [
    "TotalCharges differs from expected by 12%",
    "Potential anomalies detected in: tenure"
  ],
  "warning_count": 2
}
```

### Check Data Before Predicting

Use the `/validate` endpoint to check input quality:

```bash
curl -X POST "http://localhost:8000/validate" \
  -H "Content-Type: application/json" \
  -d '{...customer data...}'
```

For detailed validation documentation, see **[VALIDATION.md](VALIDATION.md)**.

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

## Metrics and Hyperparameter Considerations

To evaluate model performance, the project focuses on recall, precision, F1 score, and accuracy. Because the primary objective is to identify customers who are likely to churn, recall is treated as the most important metric. Missing a true churner represents a lost opportunity for retention, making false negatives more costly than false positives in this use case.

Precision is used as a secondary metric to ensure that retention efforts are not applied too broadly to customers who are unlikely to churn. F1 score is used to balance recall and precision, providing a single metric for comparing model performance. Accuracy is also recorded but is not the primary decision metric, as it can be misleading in classification problems where class imbalance exists.

For model experimentation, key hyperparameters are tracked through MLflow and defined in `params.yaml` to ensure reproducibility and consistency across runs. Logistic Regression serves as the baseline model due to its simplicity and interpretability. Decision Tree and Random Forest models are included to evaluate whether more complex, non-linear models improve performance.

Relevant hyperparameters include regularization strength and maximum iterations for Logistic Regression, tree depth and split criteria for Decision Tree, and the number of trees and depth for Random Forest. Tracking these parameters allows for better understanding of model behavior and supports future tuning efforts.

Based on experimental results, Logistic Regression achieved the strongest balance of recall and F1 score, making it the most suitable model for the current MVP. While more complex models were evaluated, they did not outperform the baseline, suggesting that the dataset relationships are effectively captured by a simpler model.

Future improvements may include hyperparameter tuning, feature engineering, and classification threshold adjustment to further improve recall and overall model performance.

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
- **DVC** - Data versioning and reproducible pipelines
- **FastAPI** - REST API framework
- **Streamlit** - Web interface
- **Uvicorn** - ASGI server

## License

This project is for educational purposes as part of Machine Learning Design for Business course.