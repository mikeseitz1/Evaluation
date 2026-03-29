
"""
California Housing Price Prediction - MLflow Experiment Tracking
Exercise 6.7 from Machine Learning Design for Business

This script demonstrates experiment tracking with MLflow using the 
California Housing dataset to predict home prices for Zillow.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def prepare_data():
    """Load and prepare the California Housing dataset."""
    print("Loading California Housing dataset...")
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    print(f"Number of features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    
    metrics = {
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred)
    }
    
    return metrics, y_pred


def train_linear_regression(X_train, X_test, y_train, y_test):
    """Train and log Linear Regression model."""
    run_name = "Linear Regression"
    
    with mlflow.start_run(run_name=run_name) as run:
        print(f"
{'='*60}")
        print(f"Training: {run_name}")
        print(f"{'='*60}")
        
        # Log dataset information
        mlflow.log_param("dataset", "California Housing")
        mlflow.log_param("train_samples", X_train.shape[0])
        mlflow.log_param("test_samples", X_test.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("preprocessing", "None")
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Log model parameters
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("fit_intercept", model.fit_intercept)
        
        # Evaluate and log metrics
        metrics, y_pred = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_test.iloc[:5]
        )
        
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"R²: {metrics['r2']:.4f}")
        
        return metrics


def train_decision_tree(X_train, X_test, y_train, y_test):
    """Train and log Decision Tree model."""
    run_name = "Decision Tree"
    
    with mlflow.start_run(run_name=run_name) as run:
        print(f"
{'='*60}")
        print(f"Training: {run_name}")
        print(f"{'='*60}")
        
        # Log dataset information
        mlflow.log_param("dataset", "California Housing")
        mlflow.log_param("train_samples", X_train.shape[0])
        mlflow.log_param("test_samples", X_test.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("preprocessing", "None")
        
        # Model hyperparameters
        params = {
            "max_depth": 10,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "random_state": 42
        }
        
        # Train model
        model = DecisionTreeRegressor(**params)
        model.fit(X_train, y_train)
        
        # Log model parameters
        mlflow.log_param("model_type", "DecisionTreeRegressor")
        mlflow.log_params(params)
        
        # Evaluate and log metrics
        metrics, y_pred = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_test.iloc[:5]
        )
        
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"R²: {metrics['r2']:.4f}")
        
        return metrics


def train_random_forest(X_train, X_test, y_train, y_test):
    """Train and log Random Forest model."""
    run_name = "Random Forest"
    
    with mlflow.start_run(run_name=run_name) as run:
        print(f"
{'='*60}")
        print(f"Training: {run_name}")
        print(f"{'='*60}")
        
        # Log dataset information
        mlflow.log_param("dataset", "California Housing")
        mlflow.log_param("train_samples", X_train.shape[0])
        mlflow.log_param("test_samples", X_test.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("preprocessing", "None")
        
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
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        
        # Log model parameters
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_params(params)
        
        # Evaluate and log metrics
        metrics, y_pred = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv('feature_importance.csv', index=False)
        mlflow.log_artifact('feature_importance.csv')
        
        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_test.iloc[:5]
        )
        
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"R²: {metrics['r2']:.4f}")
        
        return metrics


def train_random_forest_with_scaling(X_train, X_test, y_train, y_test):
    """Train and log Random Forest model with feature scaling."""
    run_name = "Random Forest with Scaling"
    
    with mlflow.start_run(run_name=run_name) as run:
        print(f"
{'='*60}")
        print(f"Training: {run_name}")
        print(f"{'='*60}")
        
        # Apply feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Log dataset information
        mlflow.log_param("dataset", "California Housing")
        mlflow.log_param("train_samples", X_train.shape[0])
        mlflow.log_param("test_samples", X_test.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("preprocessing", "StandardScaler")
        
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
        model = RandomForestRegressor(**params)
        model.fit(X_train_scaled, y_train)
        
        # Log model parameters
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_params(params)
        
        # Evaluate and log metrics
        metrics, y_pred = evaluate_model(model, X_test_scaled, y_test)
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_test_scaled[:5]
        )
        
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"R²: {metrics['r2']:.4f}")
        
        return metrics


def main():
    """Main execution function."""
    print("
" + "="*60)
    print("California Housing Price Prediction - MLflow Experiment")
    print("Exercise 6.7: Zillow Home Price Estimation")
    print("="*60)
    
    # Set experiment name
    experiment_name = "Zillow California Housing Price Prediction"
    mlflow.set_experiment(experiment_name)
    print(f"
MLflow Experiment: {experiment_name}")
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data()
    
    # Train models
    results = {}
    results['Linear Regression'] = train_linear_regression(X_train, X_test, y_train, y_test)
    results['Decision Tree'] = train_decision_tree(X_train, X_test, y_train, y_test)
    results['Random Forest'] = train_random_forest(X_train, X_test, y_train, y_test)
    results['Random Forest with Scaling'] = train_random_forest_with_scaling(X_train, X_test, y_train, y_test)
    
    # Summary
    print(f"
{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"
{'Model':<30} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
    print("-" * 60)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<30} {metrics['rmse']:<12.4f} {metrics['mae']:<12.4f} {metrics['r2']:<12.4f}")
    
    # Find best model
    best_model = min(results.items(), key=lambda x: x[1]['rmse'])
    print(f"
{'='*60}")
    print(f"Best Model: {best_model[0]}")
    print(f"RMSE: {best_model[1]['rmse']:.4f}")
    print(f"{'='*60}")
    
    print("
✅ Experiments completed successfully!")
    print("
To view results in MLflow UI, run:")
    print("  mlflow ui")
    print("
Then open: http://localhost:5000")


if __name__ == "__main__":
    main()

