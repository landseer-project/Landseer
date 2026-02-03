"""
Tests for the OOD (Out-of-Distribution) detection evaluator.

Verifies:
- Confidence score computation (1 - max_conf)
- AUC calculation
- FPR at 95% TPR calculation
- OOD sample generation
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score


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


class TestConfidenceScores:
    """Test confidence score computation."""
    
    def test_get_max_confidence_scores(self, ood_module, sample_model, data_loader):
        """Test that confidence scores are computed correctly."""
        device = "cpu"
        sample_model.to(device)
        
        scores = ood_module.get_max_confidence_scores(sample_model, data_loader, device)
        
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 50
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)
    
    def test_high_confidence_low_score(self, ood_module):
        """Test that high confidence model gives low OOD scores for ID data."""
        class ConfidentModel(nn.Module):
            def forward(self, x):
                batch_size = x.size(0)
                out = torch.zeros(batch_size, 10)
                out[:, 0] = 100.0
                return out
        
        model = ConfidentModel()
        
        X = torch.rand(20, 3, 32, 32)
        Y = torch.zeros(20).long()
        loader = DataLoader(TensorDataset(X, Y), batch_size=10)
        
        scores = ood_module.get_max_confidence_scores(model, loader, "cpu")
        
        assert np.mean(scores) < 0.1


class TestOODSampleGeneration:
    """Test OOD sample generation."""
    
    def test_generate_gaussian_ood(self, ood_module):
        """Test Gaussian OOD sample generation."""
        in_data = np.random.rand(100, 3, 32, 32).astype(np.float32)
        ood_data = ood_module.generate_ood_samples(in_data, method="gaussian")
        
        assert ood_data.shape[1:] == in_data.shape[1:]
        assert ood_data.dtype == np.float32
        assert 0.0 <= ood_data.min()
        assert ood_data.max() <= 1.0
    
    def test_generate_uniform_ood(self, ood_module):
        """Test uniform OOD sample generation."""
        in_data = np.random.rand(100, 3, 32, 32).astype(np.float32)
        ood_data = ood_module.generate_ood_samples(in_data, method="uniform")
        
        assert ood_data.shape[1:] == in_data.shape[1:]
        assert 0.0 <= ood_data.min()
        assert ood_data.max() <= 1.0
    
    def test_generate_permuted_ood(self, ood_module):
        """Test permuted OOD sample generation."""
        in_data = np.random.rand(100, 3, 32, 32).astype(np.float32)
        ood_data = ood_module.generate_ood_samples(in_data, method="permuted")
        
        assert ood_data.shape[1:] == in_data.shape[1:]


class TestFPRCalculation:
    """Test FPR at TPR calculation."""
    
    def test_calculate_fpr_at_tpr(self, ood_module):
        """Test FPR at 95% TPR calculation."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        
        fpr = ood_module.calculate_fpr_at_tpr(y_true, scores, target_tpr=0.8)
        
        assert isinstance(fpr, float)
        assert 0.0 <= fpr <= 1.0
    
    def test_fpr_worst_case(self, ood_module):
        """Test FPR returns valid value when scores are uniform."""
        y_true = np.array([0, 0, 1, 1])
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        
        fpr = ood_module.calculate_fpr_at_tpr(y_true, scores, target_tpr=0.99)
        
        assert 0.0 <= fpr <= 1.0


class TestOODMetrics:
    """Integration tests for OOD metrics."""
    
    def test_auc_calculation(self, ood_module, sample_model, data_loader):
        """Test AUC calculation with ID and OOD data."""
        device = "cpu"
        sample_model.to(device)
        
        id_scores = ood_module.get_max_confidence_scores(sample_model, data_loader, device)
        
        in_data = torch.rand(50, 3, 32, 32).numpy()
        ood_data = ood_module.generate_ood_samples(in_data, method="gaussian")
        
        ood_dataset = TensorDataset(
            torch.tensor(ood_data).float(),
            torch.zeros(len(ood_data)).long()
        )
        ood_loader = DataLoader(ood_dataset, batch_size=16)
        
        ood_scores = ood_module.get_max_confidence_scores(sample_model, ood_loader, device)
        
        all_scores = np.concatenate([id_scores, ood_scores])
        all_labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
        
        auc = roc_auc_score(all_labels, all_scores)
        
        assert isinstance(auc, float)
        assert 0.0 <= auc <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
