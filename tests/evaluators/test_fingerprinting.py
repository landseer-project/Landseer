"""
Tests for the fingerprinting evaluator.

Verifies:
- MINGD loss function
- MINGD attack perturbation generation
- Fingerprinting evaluation
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class SimpleModel(nn.Module):
    """Simple CNN model for testing."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@pytest.fixture
def sample_model():
    """Create a simple model for testing."""
    model = SimpleModel(num_classes=10)
    model.eval()
    return model


@pytest.fixture
def data_loader():
    """Create sample DataLoader."""
    X = torch.rand(50, 3, 32, 32)
    Y = torch.randint(0, 10, (50,))
    dataset = TensorDataset(X, Y)
    return DataLoader(dataset, batch_size=16, shuffle=False)


class TestMINGDLoss:
    """Test MINGD loss function."""
    
    def test_loss_mingd_returns_scalar(self, fingerprinting_module):
        """Test that loss_mingd returns a scalar tensor."""
        preds = torch.randn(8, 10)
        target = torch.randint(0, 10, (8,))
        
        loss = fingerprinting_module.loss_mingd(preds, target)
        
        assert loss.dim() == 0
        assert loss.dtype == torch.float32


class TestMINGDAttack:
    """Test MINGD attack perturbation."""
    
    def test_mingd_returns_perturbation(self, fingerprinting_module, sample_model):
        """Test that mingd returns perturbation of correct shape."""
        device = "cpu"
        sample_model.to(device)
        
        X = torch.rand(4, 3, 32, 32, requires_grad=False)
        y = torch.randint(0, 10, (4,))
        target = torch.randint(0, 10, (4,))
        
        delta = fingerprinting_module.mingd(sample_model, X, y, target, alpha=0.01, num_iter=3)
        
        assert delta.shape == X.shape
        assert not delta.requires_grad
    
    def test_mingd_bounded_perturbation(self, fingerprinting_module, sample_model):
        """Test that MINGD perturbation keeps images in valid range."""
        device = "cpu"
        sample_model.to(device)
        
        X = torch.rand(4, 3, 32, 32)
        y = torch.randint(0, 10, (4,))
        
        with torch.no_grad():
            target = sample_model(X).argmax(1)
        
        delta = fingerprinting_module.mingd(sample_model, X, y, target, alpha=0.01, num_iter=5)
        
        adv_images = X + delta
        
        assert adv_images.min() >= -0.01
        assert adv_images.max() <= 1.01


class TestFingerprintingEvaluation:
    """Test fingerprinting evaluation function."""
    
    def test_evaluate_fingerprinting_returns_tuple(self, fingerprinting_module, sample_model, data_loader):
        """Test that evaluation returns a tuple of (clean_acc, mingd_score)."""
        device = "cpu"
        sample_model.to(device)
        
        result = fingerprinting_module.evaluate_fingerprinting_mingd(sample_model, data_loader, device)
        
        # Returns tuple of (clean_accuracy, mingd_score)
        assert isinstance(result, tuple)
        assert len(result) == 2
        clean_acc, mingd_score = result
        assert isinstance(clean_acc, float)
        assert isinstance(mingd_score, float)
        assert 0.0 <= clean_acc <= 1.0
        assert 0.0 <= mingd_score <= 1.0
    
    def test_fingerprinting_with_auto_device(self, fingerprinting_module, sample_model, data_loader):
        """Test fingerprinting with automatic device detection."""
        sample_model.to("cpu")
        
        result = fingerprinting_module.evaluate_fingerprinting_mingd(sample_model, data_loader, device="cpu")
        
        assert isinstance(result, tuple)
        assert len(result) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
