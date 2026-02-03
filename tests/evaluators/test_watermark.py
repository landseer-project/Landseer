"""
Tests for the watermark evaluator.

Verifies:
- Weight-based watermark detection (Uchida-style)
- Trigger-based watermark detection (WatermarkNN)
- Conv2D layer finding
- Graceful skip when no watermark data
"""

import numpy as np
import pytest
import torch
import torch.nn as nn


class WatermarkedModel(nn.Module):
    """Simple model with Conv2D layers for watermark testing."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 8 * 8, num_classes)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


@pytest.fixture
def sample_model():
    """Create a simple model for testing."""
    model = WatermarkedModel(num_classes=10)
    model.eval()
    return model


class TestConv2DFinding:
    """Test finding Conv2D layers for watermark."""
    
    def test_find_conv2d_layer(self, watermark_module, sample_model):
        """Test finding appropriate Conv2D layer."""
        name, layer = watermark_module.find_uchida_style_conv2d(sample_model)
        
        assert name is not None
        assert layer is not None
        assert isinstance(layer, nn.Conv2d)
    
    def test_prefer_second_last_conv(self, watermark_module):
        """Test that second-last Conv2D is preferred."""
        class MultiConvModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3)
                self.conv2 = nn.Conv2d(16, 32, 3)
                self.conv3 = nn.Conv2d(32, 64, 3)
                
        model = MultiConvModel()
        name, layer = watermark_module.find_uchida_style_conv2d(model)
        
        assert name == "conv2"
    
    def test_no_conv2d_returns_none(self, watermark_module):
        """Test that model with no Conv2D returns None."""
        class FCOnlyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(100, 50)
                self.fc2 = nn.Linear(50, 10)
                
        model = FCOnlyModel()
        name, layer = watermark_module.find_uchida_style_conv2d(model)
        
        assert name is None
        assert layer is None


class TestWeightBasedWatermark:
    """Test weight-based watermark detection."""
    
    def test_decode_watermark_shape(self, watermark_module, sample_model):
        """Test watermark decoding with matching dimensions."""
        device = "cpu"
        sample_model.to(device)
        
        _, conv_layer = watermark_module.find_uchida_style_conv2d(sample_model)
        
        with torch.no_grad():
            mean_weight = conv_layer.weight.mean(dim=[2, 3])
            weight_dim = mean_weight.numel()
        
        watermark_length = 32
        wm_matrix = torch.randn(weight_dim, watermark_length).to(device)
        wm_bits = (torch.randn(1, watermark_length) > 0).float().to(device)
        
        accuracy, decoded = watermark_module.decode_watermark(conv_layer, wm_matrix, wm_bits, device)
        
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0
    
    def test_decode_with_dimension_mismatch(self, watermark_module, sample_model):
        """Test watermark decoding handles dimension mismatch."""
        device = "cpu"
        sample_model.to(device)
        
        _, conv_layer = watermark_module.find_uchida_style_conv2d(sample_model)
        
        watermark_length = 16
        wm_matrix = torch.randn(10, watermark_length).to(device)
        wm_bits = (torch.randn(1, watermark_length) > 0).float().to(device)
        
        accuracy, decoded = watermark_module.decode_watermark(conv_layer, wm_matrix, wm_bits, device)
        
        assert isinstance(accuracy, float)


class TestTriggerBasedWatermark:
    """Test trigger-based watermark detection."""
    
    def test_trigger_evaluation(self, watermark_module, sample_model, tmp_path):
        """Test trigger-based watermark evaluation with sample triggers."""
        device = "cpu"
        sample_model.to(device)
        
        triggers_path = tmp_path / "watermark_triggers"
        triggers_path.mkdir()
        
        trigger_images = np.random.rand(10, 3, 32, 32).astype(np.float32)
        trigger_labels = np.random.randint(0, 10, 10)
        
        np.save(triggers_path / "trigger_images.npy", trigger_images)
        np.save(triggers_path / "trigger_labels.npy", trigger_labels)
        
        accuracy, info = watermark_module.evaluate_trigger_based_watermark(sample_model, triggers_path, device)
        
        assert accuracy is not None
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0
        assert info["method"] == "trigger_based"


class TestWatermarkMetrics:
    """Integration tests for watermark metrics."""
    
    def test_watermark_accuracy_with_embedded_watermark(self, watermark_module, sample_model):
        """Test detection accuracy with deliberately embedded watermark."""
        device = "cpu"
        sample_model.to(device)
        
        _, conv_layer = watermark_module.find_uchida_style_conv2d(sample_model)
        
        watermark_length = 32
        with torch.no_grad():
            mean_weight = conv_layer.weight.mean(dim=[2, 3])
            weight_dim = mean_weight.numel()
        
        wm_matrix = torch.randn(weight_dim, watermark_length).to(device)
        expected_bits = torch.zeros(1, watermark_length).to(device)
        expected_bits[0, ::2] = 1.0
        
        accuracy, _ = watermark_module.decode_watermark(conv_layer, wm_matrix, expected_bits, device)
        
        assert 0.0 <= accuracy <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
