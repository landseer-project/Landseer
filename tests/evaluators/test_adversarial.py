"""
Tests for the adversarial evaluator.

Verifies:
- Clean accuracy computation
- PGD attack evaluation
- FGSM attack evaluation
- Carlini L2 attack evaluation
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
def sample_data():
    """Create sample test data."""
    X = np.random.rand(100, 3, 32, 32).astype(np.float32)
    Y = np.random.randint(0, 10, size=100)
    return X, Y


@pytest.fixture
def data_loader(sample_data):
    """Create DataLoader from sample data."""
    X, Y = sample_data
    dataset = TensorDataset(torch.tensor(X), torch.tensor(Y).long())
    return DataLoader(dataset, batch_size=32, shuffle=False)


class TestCleanAccuracy:
    """Test clean accuracy evaluation."""
    
    def test_clean_accuracy_returns_float(self, adversarial_module, sample_model, data_loader):
        """Test that clean accuracy returns a float between 0 and 1."""
        device = "cpu"
        sample_model.to(device)
        
        acc = adversarial_module.evaluate_clean_accuracy(sample_model, data_loader, device)
        
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0
    
    def test_clean_accuracy_perfect_model(self, adversarial_module):
        """Test that a perfect model achieves 100% accuracy."""
        class PerfectModel(nn.Module):
            def forward(self, x):
                batch_size = x.size(0)
                output = torch.zeros(batch_size, 10)
                output[:, 0] = 100.0
                return output
        
        model = PerfectModel()
        
        X = torch.rand(50, 3, 32, 32)
        Y = torch.zeros(50).long()
        loader = DataLoader(TensorDataset(X, Y), batch_size=16)
        
        acc = adversarial_module.evaluate_clean_accuracy(model, loader, "cpu")
        
        assert acc == 1.0


class TestPGDAttack:
    """Test PGD attack evaluation."""
    
    def test_pgd_returns_float(self, adversarial_module, sample_model, data_loader):
        """Test that PGD accuracy returns a float between 0 and 1."""
        device = "cpu"
        sample_model.to(device)
        
        acc = adversarial_module.evaluate_pgd_custom(sample_model, data_loader, device, eps=8/255, alpha=2/255, steps=3)
        
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0
    
    def test_pgd_robustness_lower_than_clean(self, adversarial_module, sample_model, data_loader):
        """Test that PGD accuracy is typically lower than or equal to clean accuracy."""
        device = "cpu"
        sample_model.to(device)
        
        clean_acc = adversarial_module.evaluate_clean_accuracy(sample_model, data_loader, device)
        pgd_acc = adversarial_module.evaluate_pgd_custom(sample_model, data_loader, device, eps=8/255, alpha=2/255, steps=3)
        
        assert pgd_acc <= clean_acc + 0.1


class TestFGSMAttack:
    """Test FGSM attack evaluation."""
    
    def test_fgsm_returns_float(self, adversarial_module, sample_model, data_loader):
        """Test that FGSM accuracy returns a float between 0 and 1."""
        device = "cpu"
        sample_model.to(device)
        
        acc = adversarial_module.evaluate_fgsm_custom(sample_model, data_loader, device, eps=8/255)
        
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0


class TestCarliniAttack:
    """Test Carlini L2 attack evaluation."""
    
    def test_carlini_attack_class(self, adversarial_module, sample_model):
        """Test CarliniL2Attack class initialization and call."""
        device = "cpu"
        sample_model.to(device)
        
        attack = adversarial_module.CarliniL2Attack(
            sample_model, device, 
            confidence=0, 
            max_iterations=10,
            binary_search_steps=1
        )
        
        images = torch.rand(4, 3, 32, 32)
        labels = torch.randint(0, 10, (4,))
        
        adv_images = attack(images, labels)
        
        assert adv_images.shape == images.shape
        assert adv_images.min() >= 0.0
        assert adv_images.max() <= 1.0
    
    def test_carlini_evaluate_returns_float(self, adversarial_module, sample_model, data_loader):
        """Test that Carlini evaluation returns a float between 0 and 1."""
        device = "cpu"
        sample_model.to(device)
        
        acc = adversarial_module.evaluate_carlini_l2(sample_model, data_loader, device, sample_size=20)
        
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0


class TestAdversarialMetrics:
    """Integration tests for adversarial metrics."""
    
    def test_all_metrics_structure(self, adversarial_module, sample_model, data_loader):
        """Test that all metrics are computed and have correct structure."""
        device = "cpu"
        sample_model.to(device)
        
        metrics = {}
        metrics["clean_accuracy"] = adversarial_module.evaluate_clean_accuracy(sample_model, data_loader, device)
        metrics["pgd_accuracy"] = adversarial_module.evaluate_pgd_custom(sample_model, data_loader, device, eps=8/255, alpha=2/255, steps=2)
        metrics["fgsm_accuracy"] = adversarial_module.evaluate_fgsm_custom(sample_model, data_loader, device, eps=8/255)
        metrics["carlini_l2_accuracy"] = adversarial_module.evaluate_carlini_l2(sample_model, data_loader, device, sample_size=20)
        
        expected_keys = ["clean_accuracy", "pgd_accuracy", "fgsm_accuracy", "carlini_l2_accuracy"]
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], float)
            assert 0.0 <= metrics[key] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
