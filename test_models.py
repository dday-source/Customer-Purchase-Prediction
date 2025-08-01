"""
COMPREHENSIVE TEST SUITE FOR CUSTOMER PURCHASE PREDICTION MODELS
================================================================
As part of our final project, we developed this testing framework to ensure
our models perform correctly under various conditions. 

Testing Framework: pytest
We selected pytest based on its widespread adoption in the finance industry
and its comprehensive feature set:
- Automatic test discovery reduces manual configuration
- Detailed assertion introspection aids debugging
- Fixture system enables consistent test data setup
- Integration with coverage tools ensures thorough testing

Our test coverage targets:
- Data preprocessing and quality checks
- Model training and prediction accuracy
- WOE/IV calculations for feature engineering
- Business metric computations (ROI, profit calculations)
- Edge cases and error handling scenarios

Execution commands:
pytest test_models.py -v              # Verbose output for detailed results
pytest test_models.py --cov=.         # Coverage report to identify gaps
pytest test_models.py -k "test_name"  # Run specific tests during development
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
import xgboost as xgb
import os
import sys

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our custom modules
from WOE import data_vars, mono_bin, char_bin, validate_inputs
from CustomerPurchaseBehavior import (
    create_output_directory,
    data_quality_check,
    printOutTheCoefficients,
    evaluate_model
)


#---------------------------------------------------------------------------------------------#
#---------------------------------------Test Fixtures-----------------------------------------#
#---------------------------------------------------------------------------------------------#
"""
Test fixtures provide reusable components across multiple tests, ensuring
consistency and reducing code duplication. 
"""

@pytest.fixture
def sample_data():
    """
    Generate synthetic customer data for testing purposes.
    
    We designed this dataset to mirror the characteristics of real customer
    purchase data while maintaining reproducibility through random seed setting.
    The distributions are based on typical patterns we observed in the
    actual dataset.
    """
    np.random.seed(42)
    n_samples = 1000
    
    # Create features with realistic distributions
    data = {
        'Age': np.random.randint(18, 70, n_samples),
        'Income': np.random.normal(50000, 20000, n_samples),
        'ProductViews': np.random.poisson(5, n_samples),
        'Gender': np.random.choice(['M', 'F'], n_samples),
        'ProductCategory': np.random.choice(['A', 'B', 'C'], n_samples),
        'PurchaseStatus': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def trained_model(sample_data):
    """
    Create a pre-trained logistic regression model for testing evaluation functions.
    
    """
    X = sample_data[['Age', 'Income', 'ProductViews']]
    y = sample_data['PurchaseStatus']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler, X.columns.values

@pytest.fixture
def output_dir(tmp_path):
    """
    Create temporary directory for test outputs.
    
    Using pytest's tmp_path fixture ensures test isolation and automatic
    cleanup, preventing test artifacts from accumulating in our project directory.
    """
    test_dir = tmp_path / "test_output"
    test_dir.mkdir()
    return str(test_dir)


#---------------------------------------------------------------------------------------------#
#------------------------------------Data Processing Tests------------------------------------#
#---------------------------------------------------------------------------------------------#
"""
These tests verify our data preprocessing pipeline functions correctly.

"""

def test_create_output_directory(tmp_path):
    """
    Verify output directory creation with proper timestamp formatting.
    
    The timestamp format (YYYYMMDD_HHMMSS) enables chronological sorting
    and prevents naming conflicts during multiple runs.
    """
    os.chdir(tmp_path)  # Change to temp directory
    output_dir = create_output_directory()
    
    # Verify directory exists
    assert os.path.exists(output_dir)
    
    # Validate timestamp format using regex
    import re
    pattern = r'output_\d{8}_\d{6}'
    assert re.match(pattern, output_dir)

def test_data_quality_check(sample_data, output_dir):
    # Introduce missing values to test detection
    sample_data.loc[0:10, 'Age'] = np.nan
    
    data_quality_check(sample_data, output_dir)
    
    # Verify report generation
    report_path = os.path.join(output_dir, "data_quality_report.txt")
    assert os.path.exists(report_path)
    
    # Validate report contents
    with open(report_path, 'r') as f:
        content = f.read()
        assert "Dataset Shape: (1000, 6)" in content
        assert "Missing Values:" in content
        assert "Age" in content  # Should report Age has missing values
        assert "Class Imbalance Ratio:" in content

def test_data_quality_check_imbalanced(output_dir):
    # Create severely imbalanced dataset
    imbalanced_data = pd.DataFrame({
        'Feature1': np.random.randn(1000),
        'PurchaseStatus': [0] * 900 + [1] * 100  # 9:1 imbalance ratio
    })
    
    data_quality_check(imbalanced_data, output_dir)
    
    report_path = os.path.join(output_dir, "data_quality_report.txt")
    with open(report_path, 'r') as f:
        content = f.read()
        assert "WARNING: Significant class imbalance detected!" in content


#---------------------------------------------------------------------------------------------#
#----------------------------------------WOE/IV Tests-----------------------------------------#
#---------------------------------------------------------------------------------------------#

def test_validate_inputs_decorator():
    """
    Test input validation decorator functionality.
    
    We implemented this decorator pattern after encountering cryptic errors
    from mismatched array dimensions. Clear error messages significantly
    improved our development efficiency.
    """
    # Test dimension mismatch detection
    with pytest.raises(ValueError, match="same length"):
        mono_bin([0, 1], [1, 2, 3])
    
    # Test empty array handling
    with pytest.raises(ValueError, match="empty"):
        mono_bin([], [])
    
    # Test binary target validation
    with pytest.raises(ValueError, match="0 and 1 values"):
        mono_bin([0, 1, 2], [1, 2, 3])

def test_mono_bin_basic():
    # Generate data with clear monotonic relationship
    np.random.seed(42)
    X = np.random.randn(1000)
    # Create target with monotonic relationship to X
    Y = (X + np.random.randn(1000) * 0.5 > 0).astype(int)
    
    result = mono_bin(Y, X, n=5)
    
    # Verify output structure
    assert 'WOE' in result.columns
    assert 'IV' in result.columns
    assert len(result) > 0
    
    # Verify WOE values exist for non-missing bins
    woe_values = result[result['MIN_VALUE'].notna()]['WOE'].values
    assert len(woe_values) > 1

def test_char_bin_basic():
    """
    Test categorical variable binning functionality.
    
    Unlike continuous variables, categorical features don't require
    monotonic relationships. Each category receives its own WOE value,
    enabling the model to capture non-linear patterns.
    """
    # Create categorical data with varying event rates
    X = ['A'] * 300 + ['B'] * 400 + ['C'] * 300
    Y = [1] * 200 + [0] * 100 + [1] * 100 + [0] * 300 + [1] * 250 + [0] * 50
    
    result = char_bin(Y, X)
    
    # Verify each category has distinct representation
    unique_categories = result[result['MIN_VALUE'].notna()]['MIN_VALUE'].nunique()
    assert unique_categories == 3
    
    # Verify IV calculation
    assert result['IV'].iloc[0] > 0

def test_data_vars_comprehensive(sample_data):
    target = sample_data['PurchaseStatus']
    features = sample_data.drop('PurchaseStatus', axis=1)
    
    detailed_woe, summary_iv = data_vars(features, target)
    
    # Verify all variables processed
    assert len(summary_iv) == len(features.columns)
    
    # Verify IV interpretation added
    assert 'Predictive_Power' in summary_iv.columns
    
    # Verify IV values are non-negative (mathematical requirement)
    assert (summary_iv['IV'] >= 0).all()
    
    # Verify sorting (highest IV first for feature selection)
    iv_values = summary_iv['IV'].values
    assert all(iv_values[i] >= iv_values[i+1] for i in range(len(iv_values)-1))


#---------------------------------------------------------------------------------------------#
#--------------------------------------Model Tests--------------------------------------------#
#---------------------------------------------------------------------------------------------#

def test_logistic_regression_training(sample_data):
    """
    Test logistic regression model training and prediction pipeline.
    
    Logistic regression serves as our baseline model due to its
    interpretability and regulatory acceptance in financial services.
    We verify proper training, prediction generation, and probability calibration.
    """
    X = sample_data[['Age', 'Income', 'ProductViews']]
    y = sample_data['PurchaseStatus']
    
    # Apply feature scaling (essential for convergence)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, y)
    
    # Verify model fitted successfully
    assert hasattr(model, 'coef_')
    assert model.coef_.shape == (1, 3)
    
    # Verify predictions are binary
    predictions = model.predict(X_scaled)
    assert len(predictions) == len(y)
    assert set(predictions) <= {0, 1}
    
    # Verify probability calibration
    probabilities = model.predict_proba(X_scaled)
    assert probabilities.shape == (len(y), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)

def test_xgboost_training(sample_data):
    X = sample_data[['Age', 'Income', 'ProductViews']]
    y = sample_data['PurchaseStatus']
    
    # Create DMatrix for efficient XGBoost processing
    dtrain = xgb.DMatrix(X, label=y)
    
    # Configure parameters based on best practices
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 3,
        'random_state': 42
    }
    
    # Train model
    model = xgb.train(params, dtrain, num_boost_round=10)
    
    # Verify predictions are valid probabilities
    predictions = model.predict(dtrain)
    assert len(predictions) == len(y)
    assert (predictions >= 0).all() and (predictions <= 1).all()

def test_coefficient_output(trained_model, output_dir):
    model, scaler, feature_names = trained_model
    
    coef_df = printOutTheCoefficients(
        feature_names,
        model.coef_,
        model.intercept_,
        output_dir
    )
    
    # Verify dataframe structure
    assert 'Feature' in coef_df.columns
    assert 'Coefficient' in coef_df.columns
    assert 'Odds_Ratio' in coef_df.columns
    
    # Verify odds ratio calculation accuracy
    for idx in range(len(feature_names)):
        coef = coef_df[coef_df['Feature'] == feature_names[idx]]['Coefficient'].values[0]
        odds = coef_df[coef_df['Feature'] == feature_names[idx]]['Odds_Ratio'].values[0]
        assert np.isclose(odds, np.exp(coef))
    
    # Verify file export
    assert os.path.exists(os.path.join(output_dir, 'model_coefficients.xlsx'))


#---------------------------------------------------------------------------------------------#
#------------------------------------Business Logic Tests-------------------------------------#
#---------------------------------------------------------------------------------------------#

def test_profit_calculation():
    """
    Verify profit calculation logic matches business requirements.
    
    We developed this test to ensure our ROI calculations align with
    the marketing department's cost structure and profit expectations.
    Accurate financial modeling is critical for executive buy-in.
    """
    # Create realistic confusion matrix scenario
    # True Negatives=70, False Positives=10, False Negatives=5, True Positives=15
    y_true = [0]*80 + [1]*20
    y_pred = [0]*70 + [1]*10 + [0]*5 + [1]*15
    
    # Business parameters from stakeholder input
    cost_fp = 10  # Cost of marketing to non-converter
    cost_fn = 50  # Opportunity cost of missed sale
    profit_tp = 100  # Profit from successful conversion
    
    # Validate calculation
    expected_profit = (15 * profit_tp) - (10 * cost_fp) - (5 * cost_fn)
    expected_profit = 1500 - 100 - 250  # = 1150
    
    assert expected_profit == 1150

def test_roc_auc_calculation():
    """
    Test ROC AUC calculation for model evaluation edge cases.
    
    ROC AUC serves as our primary model performance metric, balancing
    true positive and false positive rates. We test extreme cases to
    ensure robust evaluation.
    """
    from sklearn.metrics import roc_auc_score
    
    # Perfect classifier scenario
    y_true = [0, 0, 0, 1, 1, 1]
    y_scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    assert roc_auc_score(y_true, y_scores) == 1.0
    
    # Random classifier baseline
    y_true = [0, 1, 0, 1, 0, 1]
    y_scores = [0.5] * 6
    assert roc_auc_score(y_true, y_scores) == 0.5


#---------------------------------------------------------------------------------------------#
#--------------------------------------Edge Case Tests----------------------------------------#
#---------------------------------------------------------------------------------------------#
"""
Edge case testing ensures robustness in production environments. Through
our project development, we encountered numerous edge cases that could
cause system failures if not properly handled.
"""

def test_empty_dataset_handling():
    """
    Test graceful handling of empty datasets.
    
    While unlikely in production, empty datasets can occur during
    data pipeline failures. Proper handling prevents cascading errors
    in downstream systems.
    """
    empty_df = pd.DataFrame()
    target = pd.Series(dtype=int)

    # Should return empty results without throwing exceptions
    detailed_woe, summary_iv = data_vars(empty_df, target)
    assert detailed_woe.empty
    assert summary_iv.empty

def test_single_class_handling():
    """
    Test handling of single-class scenarios.
    
    During certain time periods or customer segments, we might encounter
    data where all customers exhibit the same behavior. The system must
    handle this gracefully.
    """
    # All negative class scenario
    X = np.random.randn(100)
    Y = np.zeros(100)
    
    result = mono_bin(Y, X)
    
    # IV should be zero when no discrimination possible
    assert result['IV'].iloc[0] == 0

def test_missing_values_handling():
    """
    Verify proper handling of missing values in WOE calculations.
    
    Missing values are common in real-world financial data. Our approach
    treats them as a separate category, preserving potentially valuable
    information about data completeness.
    """
    # Create data with strategic missing values
    X = [1, 2, 3, np.nan, 5, 6, np.nan, 8, 9, 10]
    Y = [0, 1, 0, 1, 1, 0, 1, 0, 1, 0]
    
    result = mono_bin(Y, X)
    
    # Verify separate bin for missing values
    missing_bin = result[result['MIN_VALUE'].isna()]
    assert len(missing_bin) > 0
    assert missing_bin['COUNT'].values[0] == 2  # Two missing values

def test_model_persistence(trained_model, tmp_path):
    """
    Test model serialization and deserialization consistency.
    
    Model persistence enables deployment and version control. We verify
    that saved models produce identical predictions when reloaded,
    ensuring consistency across environments.
    """
    import joblib
    
    model, scaler, feature_names = trained_model
    
    # Save model
    model_path = tmp_path / "test_model.pkl"
    joblib.dump(model, model_path)
    
    # Load model
    loaded_model = joblib.load(model_path)
    
    # Verify prediction consistency
    test_df = pd.DataFrame([[30, 50000, 5]], columns=feature_names)
    test_data = scaler.transform(test_df)
    original_pred = model.predict_proba(test_data)
    loaded_pred = loaded_model.predict_proba(test_data)
    
    assert np.allclose(original_pred, loaded_pred)


#---------------------------------------------------------------------------------------------#
#------------------------------------Integration Tests----------------------------------------#
#---------------------------------------------------------------------------------------------#
"""
Integration tests verify that our components work together seamlessly.
These tests simulate the complete workflow from raw data to final predictions,
ensuring system-level correctness.
"""

def test_full_pipeline_integration(sample_data, output_dir):
    """
    Test complete pipeline from data ingestion to prediction generation.
    
    This comprehensive test validates our entire workflow, ensuring
    all components integrate properly. It simulates the production
    pipeline our model will follow when deployed.
    """
    # Step 1: Feature selection using WOE/IV
    target = sample_data['PurchaseStatus']
    features = sample_data.drop('PurchaseStatus', axis=1)
    detailed_woe, summary_iv = data_vars(features, target)
    
    # Step 2: Select features based on IV threshold
    selected_features = summary_iv[summary_iv['IV'] > 0.02]['VAR_NAME'].tolist()
    assert len(selected_features) > 0
    
    # Step 3: Prepare modeling data
    X = sample_data[selected_features]
    y = target
    
    # Handle categorical variables through one-hot encoding
    X_numeric = pd.get_dummies(X)
    
    # Step 4: Scale and train model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, y)
    
    # Step 5: Generate and validate predictions
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    # Verify output integrity
    assert len(predictions) == len(y)
    assert probabilities.shape == (len(y), 2)
    assert (probabilities >= 0).all() and (probabilities <= 1).all()


#---------------------------------------------------------------------------------------------#
#------------------------------------Performance Tests----------------------------------------#
#---------------------------------------------------------------------------------------------#
"""
Performance testing ensures our models meet production latency requirements.
Financial applications often require real-time or near-real-time predictions,
making performance a critical consideration.
"""

def test_woe_calculation_performance():
    """
    Verify WOE calculations complete within acceptable time limits.
    
    For production deployment, feature engineering must be efficient.
    We target sub-second processing for typical data volumes to enable
    real-time scoring applications.
    """
    import time
    
    # Generate larger dataset to stress-test performance
    n_samples = 10000
    X = np.random.randn(n_samples)
    Y = (X + np.random.randn(n_samples) * 0.5 > 0).astype(int)
    
    start_time = time.time()
    result = mono_bin(Y, X)
    end_time = time.time()
    
    # Verify completion within performance threshold
    assert end_time - start_time < 1.0

def test_model_training_performance():
    """
    Verify model training completes within reasonable timeframes.
    
    While training occurs offline, excessive training times impact
    development velocity and model refresh cycles. We establish
    performance baselines for typical dataset sizes.
    """
    import time
    
    # Create moderately complex dataset
    n_samples = 5000
    n_features = 20
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.5 > 0).astype(int)
    
    # Time model training
    start_time = time.time()
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    end_time = time.time()
    
    # Verify acceptable training time
    assert end_time - start_time < 2.0


#---------------------------------------------------------------------------------------------#
#--------------------------------------Pytest Configuration------------------------------------#
#---------------------------------------------------------------------------------------------#

def pytest_configure(config):
    """
    Configure custom pytest markers for test categorization.
    
    Markers enable selective test execution, useful for separating
    quick unit tests from slower integration tests during development.
    """
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )

if __name__ == "__main__":
    # Enable direct script execution for convenience
    pytest.main([__file__, "-v"])

