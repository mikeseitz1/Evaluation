"""
Validation utilities for Telco Churn Prediction project.

Provides schema validation, outlier detection, cross-field validation,
and model performance threshold checking.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import warnings


class SchemaValidator:
    """Validates data schema and structure."""
    
    REQUIRED_COLUMNS = {
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges', 'Churn'
    }
    
    CATEGORICAL_COLUMNS = {
        'gender': ['Male', 'Female'],
        'Partner': ['Yes', 'No'],
        'Dependents': ['Yes', 'No'],
        'PhoneService': ['Yes', 'No'],
        'MultipleLines': ['Yes', 'No', 'No phone service'],
        'InternetService': ['DSL', 'Fiber optic', 'No'],
        'OnlineSecurity': ['Yes', 'No', 'No internet service'],
        'OnlineBackup': ['Yes', 'No', 'No internet service'],
        'DeviceProtection': ['Yes', 'No', 'No internet service'],
        'TechSupport': ['Yes', 'No', 'No internet service'],
        'StreamingTV': ['Yes', 'No', 'No internet service'],
        'StreamingMovies': ['Yes', 'No', 'No internet service'],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'PaperlessBilling': ['Yes', 'No'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
        'Churn': ['Yes', 'No']
    }
    
    NUMERIC_COLUMNS = {'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges'}
    
    @classmethod
    def validate_schema(cls, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that dataframe has required columns and types.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required columns
        missing_cols = cls.REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check categorical values
        for col, valid_values in cls.CATEGORICAL_COLUMNS.items():
            if col in df.columns:
                invalid_values = set(df[col].unique()) - set(valid_values)
                if invalid_values:
                    errors.append(f"Column '{col}' has invalid values: {invalid_values}")
        
        # Check numeric columns are numeric
        for col in cls.NUMERIC_COLUMNS:
            if col in df.columns:
                try:
                    pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    errors.append(f"Column '{col}' cannot be converted to numeric: {e}")
        
        # Check SeniorCitizen is binary
        if 'SeniorCitizen' in df.columns:
            unique_vals = set(df['SeniorCitizen'].unique())
            if not unique_vals.issubset({0, 1}):
                errors.append(f"SeniorCitizen must be binary (0 or 1), got: {unique_vals}")
        
        return len(errors) == 0, errors


class OutlierDetector:
    """Detects statistical outliers in data."""
    
    @staticmethod
    def detect_outliers_iqr(data: pd.Series, multiplier: float = 1.5) -> np.ndarray:
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Args:
            data: Series to check for outliers
            multiplier: IQR multiplier (1.5 is standard, 3.0 is extreme)
        
        Returns:
            Boolean array where True indicates outlier
        """
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - (multiplier * IQR)
        upper_bound = Q3 + (multiplier * IQR)
        
        return (data < lower_bound) | (data > upper_bound)
    
    @staticmethod
    def detect_outliers_zscore(data: pd.Series, threshold: float = 3.0) -> np.ndarray:
        """
        Detect outliers using Z-score method.
        
        Args:
            data: Series to check for outliers
            threshold: Z-score threshold (3.0 is standard for extreme outliers)
        
        Returns:
            Boolean array where True indicates outlier
        """
        z_scores = np.abs((data - data.mean()) / data.std())
        return z_scores > threshold
    
    @classmethod
    def check_numeric_outliers(cls, df: pd.DataFrame, columns: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Check for outliers in numeric columns.
        
        Args:
            df: DataFrame to check
            columns: Specific columns to check. If None, checks all numeric columns
        
        Returns:
            Dictionary mapping column names to outlier boolean arrays
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers = {}
        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                outliers[col] = cls.detect_outliers_iqr(df[col], multiplier=1.5)
        
        return outliers


class CrossFieldValidator:
    """Validates relationships between fields."""
    
    @staticmethod
    def validate_tenure_charges(df: pd.DataFrame, tolerance: float = 0.1) -> Tuple[List[int], List[str]]:
        """
        Validate that TotalCharges ≈ MonthlyCharges × tenure (with tolerance).
        
        Args:
            df: DataFrame to validate
            tolerance: Allowed relative difference (0.1 = 10%)
        
        Returns:
            Tuple of (invalid_indices, list_of_issues)
        """
        issues = []
        invalid_indices = []
        
        if 'TotalCharges' not in df.columns or 'MonthlyCharges' not in df.columns or 'tenure' not in df.columns:
            return invalid_indices, issues
        
        # Calculate expected total charges
        expected_total = df['MonthlyCharges'] * df['tenure']
        
        # Convert to numeric (handles any non-numeric values)
        actual_total = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Find discrepancies (allowing tolerance)
        for idx in df.index:
            if pd.isna(actual_total[idx]) or df.iloc[idx]['tenure'] == 0:
                continue
            
            expected = expected_total[idx]
            actual = actual_total[idx]
            
            if expected > 0:
                relative_diff = abs(actual - expected) / expected
                if relative_diff > tolerance:
                    invalid_indices.append(idx)
                    issues.append(
                        f"Row {idx}: TotalCharges ({actual:.2f}) differs from "
                        f"expected ({expected:.2f}) by {relative_diff*100:.1f}%"
                    )
        
        return invalid_indices, issues
    
    @staticmethod
    def validate_service_consistency(df: pd.DataFrame) -> Tuple[List[int], List[str]]:
        """
        Validate service-related fields are consistent.
        E.g., if InternetService='No', then internet-related services should be 'No internet service'.
        
        Args:
            df: DataFrame to validate
        
        Returns:
            Tuple of (invalid_indices, list_of_issues)
        """
        invalid_indices = []
        issues = []
        
        internet_dependent_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                                  'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        if 'InternetService' not in df.columns:
            return invalid_indices, issues
        
        for idx, row in df.iterrows():
            if row.get('InternetService') == 'No':
                for col in internet_dependent_cols:
                    if col in df.columns and row.get(col) != 'No internet service':
                        invalid_indices.append(idx)
                        issues.append(
                            f"Row {idx}: InternetService='No' but {col}='{row.get(col)}' "
                            f"(should be 'No internet service')"
                        )
                        break
            
            # Phone-related validation
            if row.get('PhoneService') == 'No' and 'MultipleLines' in df.columns:
                if row.get('MultipleLines') != 'No phone service':
                    invalid_indices.append(idx)
                    issues.append(
                        f"Row {idx}: PhoneService='No' but MultipleLines='{row.get('MultipleLines')}' "
                        f"(should be 'No phone service')"
                    )
        
        return invalid_indices, issues


class PerformanceValidator:
    """Validates model performance meets thresholds."""
    
    # Minimum acceptable performance metrics
    MIN_ACCURACY = 0.75
    MIN_PRECISION = 0.60
    MIN_RECALL = 0.50
    MIN_F1 = 0.55
    
    @classmethod
    def validate_performance(cls, metrics: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Validate that model metrics meet minimum thresholds.
        
        Args:
            metrics: Dictionary with keys 'accuracy', 'precision', 'recall', 'f1'
        
        Returns:
            Tuple of (passes_validation, list_of_failed_thresholds)
        """
        failures = []
        
        if metrics.get('accuracy', 0) < cls.MIN_ACCURACY:
            failures.append(f"Accuracy {metrics['accuracy']:.4f} < {cls.MIN_ACCURACY}")
        
        if metrics.get('precision', 0) < cls.MIN_PRECISION:
            failures.append(f"Precision {metrics['precision']:.4f} < {cls.MIN_PRECISION}")
        
        if metrics.get('recall', 0) < cls.MIN_RECALL:
            failures.append(f"Recall {metrics['recall']:.4f} < {cls.MIN_RECALL}")
        
        if metrics.get('f1', 0) < cls.MIN_F1:
            failures.append(f"F1 Score {metrics['f1']:.4f} < {cls.MIN_F1}")
        
        return len(failures) == 0, failures


def validate_training_data(df: pd.DataFrame, check_outliers: bool = True) -> Dict:
    """
    Comprehensive training data validation.
    
    Args:
        df: Training DataFrame
        check_outliers: Whether to check for outliers
    
    Returns:
        Dictionary with validation results
    """
    results = {
        'valid': True,
        'schema_valid': False,
        'schema_errors': [],
        'outlier_warnings': {},
        'cross_field_issues': [],
        'service_consistency_issues': []
    }
    
    # Schema validation
    schema_valid, schema_errors = SchemaValidator.validate_schema(df)
    results['schema_valid'] = schema_valid
    results['schema_errors'] = schema_errors
    
    if not schema_valid:
        results['valid'] = False
        return results
    
    # Outlier detection
    if check_outliers:
        numeric_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
        outlier_detector = OutlierDetector()
        outliers = outlier_detector.check_numeric_outliers(df, numeric_cols)
        
        for col, outlier_mask in outliers.items():
            if outlier_mask.sum() > 0:
                results['outlier_warnings'][col] = int(outlier_mask.sum())
    
    # Cross-field validation
    invalid_tenure, tenure_issues = CrossFieldValidator.validate_tenure_charges(df)
    if tenure_issues:
        results['cross_field_issues'] = tenure_issues[:5]  # Show first 5
    
    # Service consistency validation
    invalid_service, service_issues = CrossFieldValidator.validate_service_consistency(df)
    if service_issues:
        results['service_consistency_issues'] = service_issues[:5]  # Show first 5
    
    return results
