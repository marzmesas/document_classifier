import pytest
import numpy as np
from src.models.utils import generate_robustness_report


def test_generate_robustness_report_balanced():
    """Test robustness report generation with balanced classes."""
    actuals = [0, 1, 2, 3, 0, 1, 2, 3]
    predictions = [0, 1, 2, 3, 0, 1, 3, 2]  # 75% accuracy
    
    with pytest.MonkeyPatch.context() as mp:
        # Mock print to avoid output during testing
        mp.setattr('builtins.print', lambda *args, **kwargs: None)
        
        report = generate_robustness_report(actuals, predictions)
    
    # Check accuracy calculation
    assert report["Accuracy"] == 0.75
    assert 0 <= report["Precision"] <= 1
    assert 0 <= report["Recall"] <= 1
    assert 0 <= report["F1-Score"] <= 1
    
    # Check class distribution
    assert len(report["Class Distribution"]) == 4  # 4 classes
    
    # Check class 0 distribution
    assert report["Class Distribution"][0]["Actual Count"] == 2
    assert report["Class Distribution"][0]["Actual %"] == 25.0
    assert report["Class Distribution"][0]["Predicted Count"] == 2
    assert report["Class Distribution"][0]["Predicted %"] == 25.0


def test_generate_robustness_report_imbalanced():
    """Test robustness report with imbalanced classes."""
    actuals = [0, 0, 0, 0, 0, 1, 2, 3]  # Class 0 dominates
    predictions = [0, 0, 0, 0, 1, 1, 2, 3]  # 87.5% accuracy
    
    with pytest.MonkeyPatch.context() as mp:
        # Mock print to avoid output during testing
        mp.setattr('builtins.print', lambda *args, **kwargs: None)
        
        report = generate_robustness_report(actuals, predictions)
    
    # Check accuracy calculation
    assert report["Accuracy"] == 0.875
    
    # Check class distribution for dominant class
    assert report["Class Distribution"][0]["Actual Count"] == 5
    assert report["Class Distribution"][0]["Actual %"] == 62.5
    assert report["Class Distribution"][0]["Predicted Count"] == 4
    assert report["Class Distribution"][0]["Predicted %"] == 50.0


def test_generate_robustness_report_perfect_predictions():
    """Test robustness report with perfect predictions."""
    actuals = [0, 1, 2, 3, 4]
    predictions = [0, 1, 2, 3, 4]  # 100% accuracy
    
    with pytest.MonkeyPatch.context() as mp:
        # Mock print to avoid output during testing
        mp.setattr('builtins.print', lambda *args, **kwargs: None)
        
        report = generate_robustness_report(actuals, predictions)
    
    # Check metrics (should all be perfect)
    assert report["Accuracy"] == 1.0
    assert report["Precision"] == 1.0
    assert report["Recall"] == 1.0
    assert report["F1-Score"] == 1.0


def test_generate_robustness_report_all_wrong():
    """Test robustness report with all incorrect predictions."""
    actuals = [0, 1, 2, 3]
    predictions = [1, 2, 3, 0]  # 0% accuracy
    
    with pytest.MonkeyPatch.context() as mp:
        # Mock print to avoid output during testing
        mp.setattr('builtins.print', lambda *args, **kwargs: None)
        
        report = generate_robustness_report(actuals, predictions)
    
    # Check metrics (should all be terrible)
    assert report["Accuracy"] == 0.0
    assert report["Precision"] == 0.0
    assert report["Recall"] == 0.0
    assert report["F1-Score"] == 0.0


def test_generate_robustness_report_single_class():
    """Test robustness report with a single class."""
    actuals = [1, 1, 1, 1]
    predictions = [1, 1, 1, 1]
    
    with pytest.MonkeyPatch.context() as mp:
        # Mock print to avoid output during testing
        mp.setattr('builtins.print', lambda *args, **kwargs: None)
        
        report = generate_robustness_report(actuals, predictions)
    
    # Check metrics
    assert report["Accuracy"] == 1.0
    assert report["Precision"] == 1.0
    assert report["Recall"] == 1.0
    assert report["F1-Score"] == 1.0
    
    # Check that there is only one class in the distribution
    assert len(report["Class Distribution"]) == 1
    assert 1 in report["Class Distribution"]


def test_generate_robustness_report_empty_input():
    """Test robustness report with empty inputs."""
    actuals = []
    predictions = []
    
    with pytest.MonkeyPatch.context() as mp:
        # Mock print to avoid output during testing
        mp.setattr('builtins.print', lambda *args, **kwargs: None)
        
        # The function returns NaN values with empty inputs rather than raising an exception
        report = generate_robustness_report(actuals, predictions)
        
        # Verify we get appropriate results for empty inputs
        assert 'Accuracy' in report
        assert 'Precision' in report
        assert 'Recall' in report
        assert 'F1-Score' in report
        
        # Check for NaN values which would result from empty inputs
        import math
        assert math.isnan(report['Accuracy']) or report['Accuracy'] == 0
        
        # Class distribution should be empty
        assert isinstance(report['Class Distribution'], dict)
        assert len(report['Class Distribution']) == 0


def test_generate_robustness_report_numpy_arrays():
    """Test robustness report with numpy array inputs."""
    actuals = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    predictions = np.array([0, 1, 2, 3, 0, 1, 3, 2])  # 75% accuracy
    
    with pytest.MonkeyPatch.context() as mp:
        # Mock print to avoid output during testing
        mp.setattr('builtins.print', lambda *args, **kwargs: None)
        
        report = generate_robustness_report(actuals, predictions)
    
    # Check accuracy calculation
    assert report["Accuracy"] == 0.75


def test_generate_robustness_report_mismatched_lengths():
    """Test robustness report with inputs of different lengths."""
    actuals = [0, 1, 2, 3, 4]
    predictions = [0, 1, 2, 3]  # Missing one prediction
    
    with pytest.MonkeyPatch.context() as mp:
        # Mock print to avoid output during testing
        mp.setattr('builtins.print', lambda *args, **kwargs: None)
        
        with pytest.raises(Exception):
            # Should raise exception when inputs have different lengths
            generate_robustness_report(actuals, predictions) 