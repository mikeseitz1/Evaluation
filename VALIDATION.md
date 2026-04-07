# Validation Guide for Telco Churn Prediction Project

## Overview

This project includes comprehensive validation at three levels:

1. **Training Data Validation** - Ensures data quality during model training
2. **API Input Validation** - Validates prediction requests with cross-field checks and anomaly detection
3. **Model Performance Validation** - Ensures trained models meet quality thresholds

---

## 1. Training Data Validation (`run2.py`)

### Schema Validation
Validates that the training dataset has the required structure:
- **Required Columns**: All 20 expected columns present
- **Data Types**: Numeric columns contain proper numbers, categorical columns contain allowed values
- **Categorical Values**: Each categorical field is restricted to valid values
  - Example: `gender` must be `'Male'` or `'Female'`
  - Example: `Contract` must be one of: `'Month-to-month'`, `'One year'`, `'Two year'`

### Data Quality Checks

**Outlier Detection** (IQR Method)
- Detects statistical anomalies in numeric columns: tenure, MonthlyCharges, TotalCharges
- Uses Interquartile Range (IQR) with 1.5x multiplier
- Reports warnings for unusual values that may skew models
- Example output:
  ```
  ⚠️  Outliers detected in numeric columns:
    - tenure: 45 outliers
    - TotalCharges: 23 outliers
  ```

**Cross-Field Validation**
- Checks logical relationships between fields
- **Tenure vs Charges**: TotalCharges should approximately equal MonthlyCharges × tenure (within 10% tolerance)
- Example issue:
  ```
  TotalCharges differs from expected by 45.2%
  ```

**Service Consistency Validation**
- Validates that service flags are logically consistent
- If `InternetService = 'No'`, then all internet-dependent services must be `'No internet service'`
- If `PhoneService = 'No'`, then `MultipleLines` must be `'No phone service'`

### Model Performance Thresholds

Trained models are **only logged** to MLflow if they meet minimum performance standards:

| Metric | Minimum Threshold |
|--------|------------------|
| Accuracy | 0.75 (75%) |
| Precision | 0.60 (60%) |
| Recall | 0.50 (50%) |
| F1 Score | 0.55 (55%) |

If a model fails validation, it is logged with `validation_status: FAILED` tag but the serialized model is NOT saved.

---

## 2. API Input Validation (`app.py`)

### Type & Format Validation (via Pydantic)
Input is validated through the `ChurnPredictionInput` model:

```python
class ChurnPredictionInput(BaseModel):
    gender: Literal['Male', 'Female']  # Restricted values
    tenure: int  # Integer type
    MonthlyCharges: float  # Floating point
    TotalCharges: float  # Floating point
    # ... etc
```

### Cross-Field Validation
Checks for logical inconsistencies between fields:

**1. Tenure vs Charges Consistency**
```
Expected: TotalCharges ≈ MonthlyCharges × tenure
Tolerance: 10%
Warning if: Actual TotalCharges differs by > 10%
```

Example:
- Input: tenure=24 months, MonthlyCharges=$50, TotalCharges=$500
- Expected: $50 × 24 = $1200
- Actual: $500
- Difference: 58% → **Warning triggered**

**2. Service Consistency**
- If `InternetService = 'No'`, warns if any internet service (Security, Backup, Streaming, etc.) is enabled
- If `PhoneService = 'No'`, warns if `MultipleLines` is not `'No phone service'`

### Outlier & Anomaly Detection

Flags unusual input values using Interquartile Range (IQR) method:

- Checks: `tenure`, `MonthlyCharges`, `TotalCharges`
- Flags values outside expected distribution
- Warning message:
  ```
  Potential anomalies detected in: tenure, TotalCharges
  Prediction may be less reliable for unusual input values.
  ```

### New API Endpoints

#### `/validate` (POST)
Validates input WITHOUT making a prediction

```bash
curl -X POST "http://localhost:8000/validate" \
  -H "Content-Type: application/json" \
  -d '{"gender": "Male", "tenure": 500, ...}'
```

Response:
```json
{
  "valid": false,
  "validation_warnings": [
    "TotalCharges differs from expected by 45.2%",
    "Potential anomalies detected in: tenure"
  ],
  "warning_count": 2,
  "message": "Found 2 validation issue(s)"
}
```

#### `/predict` (POST)
Makes prediction and includes validation warnings in response

Response now includes:
```json
{
  "prediction": "Churn",
  "churn_probability": 0.72,
  "confidence": 0.72,
  "validation_warnings": [
    "TotalCharges differs from expected by 45.2%"
  ],
  "warning_count": 1
}
```

#### `/health` (GET)
Enhanced with validation info:

```json
{
  "status": "healthy",
  "model": "Logistic Regression",
  "features": 28,
  "validation": "Enabled",
  "checks": ["schema", "cross_field", "outliers"]
}
```

---

## 3. Validation Utility Module (`validation.py`)

### Classes

#### `SchemaValidator`
Validates data structure and types
```python
valid, errors = SchemaValidator.validate_schema(df)
# Returns: (bool, list_of_error_messages)
```

#### `OutlierDetector`
Detects statistical anomalies using multiple methods

**IQR Method** (for normal distributions):
```python
outliers = OutlierDetector.detect_outliers_iqr(series, multiplier=1.5)
```

**Z-Score Method** (for extreme outliers):
```python
outliers = OutlierDetector.detect_outliers_zscore(series, threshold=3.0)
```

#### `CrossFieldValidator`
Validates relationships between fields

**Tenure vs Charges**:
```python
invalid_indices, issues = CrossFieldValidator.validate_tenure_charges(df, tolerance=0.1)
```

**Service Consistency**:
```python
invalid_indices, issues = CrossFieldValidator.validate_service_consistency(df)
```

#### `PerformanceValidator`
Validates model metrics against thresholds

```python
passes, failures = PerformanceValidator.validate_performance(metrics)
# Example failure:
# "Precision 0.5432 < 0.6"
```

### Helper Function

#### `validate_training_data()`
Comprehensive training data validation
```python
results = validate_training_data(df, check_outliers=True)
# Returns detailed report with schema, outliers, cross-field issues
```

---

## Usage Examples

### Training with Validation
```bash
python run2.py
```

Output shows:
```
Validating data schema...
✓ Schema validation passed

Checking data quality...
⚠️  Outliers detected in numeric columns:
  - tenure: 45 outliers
⚠️  Cross-field validation issues (showing first 5):
  - Row 234: TotalCharges differs from expected by 12.3%

Training: Logistic Regression
...
✓ Model passes performance thresholds
```

### API Usage with Validation

**Check validation before predicting**:
```bash
curl -X POST "http://localhost:8000/validate" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "tenure": 2,
    "MonthlyCharges": 85.5,
    "TotalCharges": 2045.0,
    ...
  }'
```

**Get prediction with validation warnings**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{...}'
```

---

## Configuration

### Adjusting Performance Thresholds

Edit `validation.py` class `PerformanceValidator`:
```python
class PerformanceValidator:
    MIN_ACCURACY = 0.75  # Change to 0.80 for stricter
    MIN_PRECISION = 0.60
    MIN_RECALL = 0.50
    MIN_F1 = 0.55
```

### Adjusting Outlier Detection Sensitivity

In `validation.py`, modify multipliers:
```python
# More sensitive (detects more outliers):
outliers = detect_outliers_iqr(data, multiplier=1.0)

# Less sensitive (fewer false positives):
outliers = detect_outliers_iqr(data, multiplier=2.0)
```

### Adjusting Cross-Field Tolerance

In `app.py`, modify tolerance in `preprocess_input()`:
```python
if relative_diff > 0.1:  # 10% - change to 0.05 for stricter
    warnings.append(...)
```

---

## When Predictions Are Unreliable

Predictions should be treated with caution if:
- ✓ `validation_warning_count > 0`
- ✓ Any outliers detected
- ✓ Service consistency issues
- ✓ Cross-field validation failures

The API returns warnings in responses, allowing you to decide whether to use the prediction.

---

## Next Steps

1. **Monitor validation warnings** in production - they indicate potential data quality issues
2. **Adjust thresholds** based on your domain knowledge
3. **Log warnings** to track systematic data quality problems
4. **Consider retraining** if warnings become frequent
5. **Share validation results** with data quality teams

---

## Integration with MLflow

Train runs are labeled with:
- `validation_status: PASSED` - Model meets all thresholds
- `validation_status: FAILED` - Model fails thresholds (not deployed)

View in MLflow UI to filter by validation status:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
