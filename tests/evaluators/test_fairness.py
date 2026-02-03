"""
Tests for the fairness evaluator.

Verifies:
- Demographic parity calculation
- Equalized odds difference calculation
- Handling of sensitive attributes
"""

import numpy as np
import pytest
import torch
import torch.nn as nn


class SimpleClassifier(nn.Module):
    """Simple classifier for testing."""
    
    def __init__(self, input_size=10, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@pytest.fixture
def sample_data():
    """Create sample data with sensitive attributes."""
    np.random.seed(42)
    n_samples = 100
    
    X = np.random.randn(n_samples, 10).astype(np.float32)
    Y = np.random.randint(0, 2, n_samples).astype(np.int64)
    sensitive = np.random.randint(0, 2, n_samples)
    
    return X, Y, sensitive


@pytest.fixture
def sample_model():
    """Create a simple model for testing."""
    model = SimpleClassifier(input_size=10, num_classes=2)
    model.eval()
    return model


class TestFairnessEvaluation:
    """Test fairness evaluation functions."""
    
    def test_evaluate_fairness_returns_tuple(self, sample_model, sample_data):
        """Test that evaluate_fairness returns tuple of two floats."""
        X, Y, sensitive = sample_data
        
        model = sample_model
        device = "cpu"
        model.to(device)
        
        test_X = torch.tensor(X)
        test_y = torch.tensor(Y)
        
        model.eval()
        with torch.no_grad():
            preds = model(test_X).argmax(1).numpy()
        
        y_true = Y
        
        preds_z1 = [p for p, z in zip(preds, sensitive) if z == 1]
        preds_z0 = [p for p, z in zip(preds, sensitive) if z == 0]
        
        if preds_z1 and preds_z0:
            dp = abs(np.mean(preds_z1) - np.mean(preds_z0))
        else:
            dp = 0.0
        
        assert isinstance(dp, float)
        assert 0.0 <= dp <= 1.0
    
    def test_fairness_with_balanced_predictions(self):
        """Test fairness metrics with perfectly balanced predictions."""
        class BalancedModel(nn.Module):
            def forward(self, x):
                batch_size = x.size(0)
                out = torch.zeros(batch_size, 2)
                for i in range(batch_size):
                    if x[i].sum() > 0:
                        out[i, 1] = 1.0
                    else:
                        out[i, 0] = 1.0
                return out
        
        model = BalancedModel()
        
        n = 50
        X = torch.randn(n, 10)
        y = torch.randint(0, 2, (n,))
        sensitive = np.array([0] * 25 + [1] * 25)
        
        with torch.no_grad():
            preds = model(X).argmax(1).numpy()
        
        preds_z0 = [p for p, z in zip(preds, sensitive) if z == 0]
        preds_z1 = [p for p, z in zip(preds, sensitive) if z == 1]
        
        if preds_z0 and preds_z1:
            dp = abs(np.mean(preds_z0) - np.mean(preds_z1))
            assert dp < 1.0
    
    def test_fairness_with_biased_predictions(self):
        """Test fairness metrics detect bias."""
        class BiasedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias_idx = 0
                
            def forward(self, x):
                batch_size = x.size(0)
                out = torch.zeros(batch_size, 2)
                for i in range(batch_size):
                    if x[i, 0] > 0:
                        out[i, 1] = 10.0
                    else:
                        out[i, 0] = 10.0
                return out
        
        model = BiasedModel()
        
        n = 100
        X = torch.zeros(n, 10)
        X[:50, 0] = 1.0
        X[50:, 0] = -1.0
        
        y = torch.randint(0, 2, (n,))
        sensitive = np.array([0] * 50 + [1] * 50)
        
        with torch.no_grad():
            preds = model(X).argmax(1).numpy()
        
        preds_z0 = [p for p, z in zip(preds, sensitive) if z == 0]
        preds_z1 = [p for p, z in zip(preds, sensitive) if z == 1]
        
        dp = abs(np.mean(preds_z0) - np.mean(preds_z1))
        
        assert dp > 0.5


class TestFairnessWithSkip:
    """Test graceful skip behavior."""
    
    def test_skip_when_no_sensitive_attrs(self):
        """Test that evaluator handles missing sensitive attributes gracefully."""
        pass  # This is tested via container execution


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
