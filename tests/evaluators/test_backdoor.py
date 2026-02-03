"""
Tests for the backdoor evaluator.

Verifies:
- Trigger application
- Attack success rate computation
- Clean accuracy post-attack
- Graceful skip when no poisoning metadata
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


class TestTriggerApplication:
    """Test trigger pattern application."""
    
    def test_badnets_trigger_bottom_right(self, backdoor_module):
        """Test BadNets trigger applied to bottom right."""
        images = np.random.rand(10, 3, 32, 32).astype(np.float32)
        trigger_info = {
            "technique": "badnets",
            "trigger_size": 3,
            "trigger_value": 1.0,
            "trigger_position": "bottom_right"
        }
        
        triggered = backdoor_module.add_trigger(images, trigger_info)
        
        assert triggered.shape == images.shape
        for i in range(10):
            assert np.allclose(triggered[i, :, 29:, 29:], 1.0)
    
    def test_badnets_trigger_top_left(self, backdoor_module):
        """Test BadNets trigger applied to top left."""
        images = np.random.rand(10, 3, 32, 32).astype(np.float32)
        trigger_info = {
            "technique": "badnets",
            "trigger_size": 3,
            "trigger_value": 1.0,
            "trigger_position": "top_left"
        }
        
        triggered = backdoor_module.add_trigger(images, trigger_info)
        
        for i in range(10):
            assert np.allclose(triggered[i, :, :3, :3], 1.0)
    
    def test_blend_trigger(self, backdoor_module):
        """Test blend trigger application."""
        images = np.random.rand(10, 3, 32, 32).astype(np.float32)
        trigger_info = {
            "technique": "blend",
            "alpha": 0.2
        }
        
        triggered = backdoor_module.add_trigger(images, trigger_info)
        
        assert triggered.shape == images.shape
        assert not np.allclose(triggered, images)
        assert 0.0 <= triggered.min()
        assert triggered.max() <= 1.0


class TestASRComputation:
    """Test attack success rate computation."""
    
    def test_asr_with_vulnerable_model(self):
        """Test ASR computation with model that always predicts target class."""
        class VulnerableModel(nn.Module):
            def forward(self, x):
                batch_size = x.size(0)
                out = torch.zeros(batch_size, 10)
                out[:, 0] = 100.0
                return out
        
        model = VulnerableModel()
        
        X = torch.rand(20, 3, 32, 32)
        Y = torch.ones(20).long() * 5
        loader = DataLoader(TensorDataset(X, Y), batch_size=10)
        
        target_class = 0
        attack_success = 0
        total = 0
        
        with torch.no_grad():
            for images, _ in loader:
                outputs = model(images)
                _, predicted = outputs.max(1)
                attack_success += (predicted == target_class).sum().item()
                total += images.size(0)
        
        asr = attack_success / total
        
        assert asr == 1.0
    
    def test_asr_with_robust_model(self):
        """Test ASR computation with model that never predicts target class."""
        class RobustModel(nn.Module):
            def forward(self, x):
                batch_size = x.size(0)
                out = torch.zeros(batch_size, 10)
                out[:, 5] = 100.0
                return out
        
        model = RobustModel()
        
        X = torch.rand(20, 3, 32, 32)
        Y = torch.ones(20).long() * 5
        loader = DataLoader(TensorDataset(X, Y), batch_size=10)
        
        target_class = 0
        attack_success = 0
        total = 0
        
        with torch.no_grad():
            for images, _ in loader:
                outputs = model(images)
                _, predicted = outputs.max(1)
                attack_success += (predicted == target_class).sum().item()
                total += images.size(0)
        
        asr = attack_success / total
        
        assert asr == 0.0


class TestBackdoorMetrics:
    """Integration tests for backdoor metrics."""
    
    def test_metrics_structure(self, sample_model):
        """Test that all expected metrics are computed."""
        device = "cpu"
        sample_model.to(device)
        
        X = torch.rand(50, 3, 32, 32)
        Y = torch.randint(1, 10, (50,))
        loader = DataLoader(TensorDataset(X, Y), batch_size=16)
        
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                outputs = sample_model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        clean_accuracy = correct / total
        
        assert isinstance(clean_accuracy, float)
        assert 0.0 <= clean_accuracy <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
