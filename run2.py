
"""
Telco Customer Churn Prediction - MLflow Experiment Tracking
Exercise 6.7 from Machine Learning Design for Business

This script demonstrates experiment tracking with MLflow using the 
Telco Customer Churn dataset to predict customer churn.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from validation import SchemaValidator, OutlierDetector, CrossFieldValidator, PerformanceValidator, validate_training_data


def prepare_data():
    """Load and prepare the Telco Customer Churn dataset with validation."""
    print("Loading Telco Customer Churn dataset...")
    data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Validate schema before processing
    print("\nValidating data schema...")
    schema_valid, schema_errors = SchemaValidator.validate_schema(data)
    if not schema_valid:
        print("⚠️  Schema validation warnings:")
        for error in schema_errors:
            print(f"  - {error}")
    else:
        print("✓ Schema validation passed")
    
    # Check for data quality issues
    print("\nChecking data quality...")
    validation_results = validate_training_data(data, check_outliers=True)
    
    if validation_results['outlier_warnings']:
        print("⚠️  Outliers detected in numeric columns:")
        for col, count in validation_results['outlier_warnings'].items():
            print(f"  - {col}: {count} outliers")
    
    if validation_results['cross_field_issues']:
        print("⚠️  Cross-field validation issues (showing first 5):")
        for issue in validation_results['cross_field_issues']:
            print(f"  - {issue}")
    
    if validation_results['service_consistency_issues']:
        print("⚠️  Service consistency issues (showing first 5):")
        for issue in validation_results['service_consistency_issues']:
            print(f"  - {issue}")
    
    # Drop customerID as it's not useful for prediction
    data = data.drop('customerID', axis=1)
    
    # Convert TotalCharges to numeric, handling empty strings
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    
    # Impute missing TotalCharges with mean
    data['TotalCharges'].fillna(data['TotalCharges'].mean(), inplace=True)
    
    # Convert Churn to binary: Yes=1, No=0
    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
    
    # Separate features and target
    X = data.drop('Churn', axis=1)
    y = data['Churn']
    
    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Save preprocessing artifacts for API use
    import joblib
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(list(X.columns), 'feature_columns.pkl')
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    print(f"Number of features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }
    
    return metrics, y_pred


def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Train and log Logistic Regression model with tuning."""
    
    best_metrics = None

    for c_value in [0.5, 1.0, 2.0]:
        run_name = f"Logistic Regression C={c_value}"

        with mlflow.start_run(run_name=run_name):
            print(f"{'='*60}")
            print(f"Training: {run_name}")
            print(f"{'='*60}")

            # Log dataset info
            mlflow.log_param("dataset", "Telco Customer Churn")
            mlflow.log_param("train_samples", X_train.shape[0])
            mlflow.log_param("test_samples", X_test.shape[0])
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("preprocessing", "StandardScaler + OneHotEncoding + Imputation")

            # 🔑 Updated model (THIS is the important part)
            model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced',
                C=c_value
            )
            model.fit(X_train, y_train)

            # Log parameters
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("random_state", 42)
            mlflow.log_param("max_iter", 1000)
            mlflow.log_param("class_weight", "balanced")
            mlflow.log_param("C", c_value)

            # Evaluate
            metrics, y_pred = evaluate_model(model, X_test, y_test)
            mlflow.log_metrics(metrics)

            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")

            # Validation check
            passes_validation, failures = PerformanceValidator.validate_performance(metrics)
            if passes_validation:
                print("✓ Model passes performance thresholds")
                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                    input_example=X_test.iloc[:5]
                )
            else:
                print("⚠️ Model does NOT meet performance thresholds:")
                for failure in failures:
                    print(f"  - {failure}")
                mlflow.log_param("validation_status", "FAILED")

            # Track best model
            if best_metrics is None or metrics['f1'] > best_metrics['f1']:
                best_metrics = metrics

    return best_metrics


def train_decision_tree(X_train, X_test, y_train, y_test):
    """Train and log Decision Tree model."""
    run_name = "Decision Tree"
    
    with mlflow.start_run(run_name=run_name) as run:
        print("="*60)
        print(f"Training: {run_name}")
        print("="*60)
        
        # Log dataset information
        mlflow.log_param("dataset", "Telco Customer Churn")
        mlflow.log_param("train_samples", X_train.shape[0])
        mlflow.log_param("test_samples", X_test.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("preprocessing", "StandardScaler + OneHotEncoding + Imputation")
        
        # Model hyperparameters
        params = {
            "max_depth": 10,
            "min_samples_split": 25,
            "min_samples_leaf": 15,
            "random_state": 42
        }
        
        # Train model
        model = DecisionTreeClassifier(**params)
        model.fit(X_train, y_train)
        
        # Log model parameters
        mlflow.log_param("model_type", "DecisionTreeClassifier")
        mlflow.log_params(params)
        
        # Evaluate and log metrics
        metrics, y_pred = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        
        # Validate performance thresholds
        passes_validation, failures = PerformanceValidator.validate_performance(metrics)
        if passes_validation:
            print("✓ Model passes performance thresholds")
            # Log model only if it passes validation
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                input_example=X_test.iloc[:5]
            )
        else:
            print("⚠️  Model does NOT meet performance thresholds:")
            for failure in failures:
                print(f"  - {failure}")
            mlflow.log_param("validation_status", "FAILED")
        
        return metrics


def train_random_forest(X_train, X_test, y_train, y_test):
    """Train and log Random Forest model."""
    run_name = "Random Forest"
    
    with mlflow.start_run(run_name=run_name) as run:
        print("="*60)
        print(f"Training: {run_name}")
        print("="*60)
        
        # Log dataset information
        mlflow.log_param("dataset", "Telco Customer Churn")
        mlflow.log_param("train_samples", X_train.shape[0])
        mlflow.log_param("test_samples", X_test.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("preprocessing", "StandardScaler + OneHotEncoding + Imputation")
        
        # Model hyperparameters
        params = {
            "n_estimators": 100,
            "max_depth": 15,
            "min_samples_split": 10,
            "min_samples_leaf": 4,
            "random_state": 42,
            "n_jobs": -1
        }
        
        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Log model parameters
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_params(params)
        
        # Evaluate and log metrics
        metrics, y_pred = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        csv_path = os.path.join(os.getcwd(), 'feature_importance.csv')
        feature_importance.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        
        # Validate performance thresholds
        passes_validation, failures = PerformanceValidator.validate_performance(metrics)
        if passes_validation:
            print("✓ Model passes performance thresholds")
            # Log model only if it passes validation
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                input_example=X_test.iloc[:5]
            )
        else:
            print("⚠️  Model does NOT meet performance thresholds:")
            for failure in failures:
                print(f"  - {failure}")
            mlflow.log_param("validation_status", "FAILED")
        
        return metrics



def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("Telco Customer Churn Prediction - MLflow Experiment")
    print("Exercise 6.7: Customer Churn Classification")
    print("="*60)
    
    # Set MLflow tracking URI to local SQLite database
    tracking_uri = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow Tracking URI: {tracking_uri}")
    
    # Set experiment name
    experiment_name = "Telco Customer Churn Prediction"
    mlflow.set_experiment(experiment_name)
    print(f"MLflow Experiment: {experiment_name}")
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data()
    
    # Train models
    results = {}
    results['Logistic Regression'] = train_logistic_regression(X_train, X_test, y_train, y_test)
    results['Decision Tree'] = train_decision_tree(X_train, X_test, y_train, y_test)
    results['Random Forest'] = train_random_forest(X_train, X_test, y_train, y_test)
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"\n{'Model':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 60)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<30} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['f1'])
    print("\n" + "="*60)
    print(f"Best Model: {best_model[0]}")
    print(f"F1 Score: {best_model[1]['f1']:.4f}")
    print("="*60)
    
    print("\n[SUCCESS] Experiments completed successfully!")
    print("\nTo view results in MLflow UI, run:")
    print("  mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print("\nThen open: http://localhost:5000")


if __name__ == "__main__":
    main()
