"""
Tests for the poisoning module.

Verifies:
- All poisoning strategies work correctly
- Metadata is properly recorded
- Poisoned indices are tracked
- Validation checks work
"""

import numpy as np
import pytest

from src.data.poisoning import (
    PoisoningStrategy,
    PoisoningResult,
    PoisoningMetadata,
    PoisonType,
    AttackGoal,
    get_poisoning_strategy,
    list_poisoning_strategies,
)


@pytest.fixture
def sample_data():
    """Create sample dataset for testing."""
    np.random.seed(42)
    n_samples = 100
    
    # CIFAR-like: 100 samples, 3 channels, 32x32
    data = np.random.rand(n_samples, 3, 32, 32).astype(np.float32)
    labels = np.random.randint(0, 10, n_samples)
    
    return data, labels


class TestPoisoningRegistry:
    """Test poisoning strategy registry."""
    
    def test_list_strategies(self):
        """Test listing available strategies."""
        strategies = list_poisoning_strategies()
        
        assert "badnets" in strategies
        assert "blend" in strategies
        assert "wanet" in strategies
        assert "label_flip" in strategies
    
    def test_get_badnets_strategy(self):
        """Test getting BadNets strategy."""
        strategy = get_poisoning_strategy("badnets")
        
        assert strategy is not None
        assert strategy.name == "badnets"
        assert strategy.poison_type == PoisonType.BACKDOOR
        assert strategy.requires_trigger == True
    
    def test_get_label_flip_strategy(self):
        """Test getting label-flip strategy."""
        strategy = get_poisoning_strategy("label_flip")
        
        assert strategy is not None
        assert strategy.name == "label_flip"
        assert strategy.poison_type == PoisonType.LABEL_FLIP
        assert strategy.requires_trigger == False
    
    def test_get_unknown_strategy(self):
        """Test getting unknown strategy returns None."""
        strategy = get_poisoning_strategy("unknown_strategy")
        assert strategy is None


class TestBadNetsStrategy:
    """Test BadNets backdoor attack strategy."""
    
    def test_basic_poisoning(self, sample_data):
        """Test basic BadNets poisoning."""
        data, labels = sample_data
        strategy = get_poisoning_strategy("badnets")
        
        result = strategy.apply(
            data=data,
            labels=labels,
            dataset_id="test_cifar10",
            poison_rate=0.1,
            target_class=0,
            seed=42,
        )
        
        assert isinstance(result, PoisoningResult)
        assert result.data.shape == data.shape
        assert len(result.labels) == len(labels)
        assert result.metadata.num_poisoned > 0
        assert result.metadata.poison_rate_actual <= 0.1 + 0.01  # Allow small tolerance
    
    def test_poisoned_indices_tracked(self, sample_data):
        """Test that poisoned indices are properly tracked."""
        data, labels = sample_data
        strategy = get_poisoning_strategy("badnets")
        
        result = strategy.apply(
            data=data,
            labels=labels,
            dataset_id="test",
            poison_rate=0.1,
            target_class=0,
            seed=42,
        )
        
        # Check indices are recorded
        assert len(result.metadata.poisoned_indices) == result.metadata.num_poisoned
        
        # Check indices are sorted
        indices = list(result.metadata.poisoned_indices)
        assert indices == sorted(indices)
        
        # Check indices are unique
        assert len(indices) == len(set(indices))
        
        # Check indices are in bounds
        assert all(0 <= idx < len(data) for idx in indices)
    
    def test_trigger_applied_correctly(self, sample_data):
        """Test that trigger is applied to poisoned samples."""
        data, labels = sample_data
        strategy = get_poisoning_strategy("badnets")
        
        result = strategy.apply(
            data=data,
            labels=labels,
            dataset_id="test",
            poison_rate=0.1,
            target_class=0,
            trigger_size=3,
            trigger_value=1.0,
            trigger_position="bottom_right",
            seed=42,
        )
        
        # Check trigger is present in poisoned samples
        for idx in result.metadata.poisoned_indices:
            # Bottom-right corner should have trigger value
            assert np.allclose(result.data[idx, :, 29:, 29:], 1.0)
            # Label should be target class
            assert result.labels[idx] == 0
    
    def test_apply_trigger_for_evaluation(self, sample_data):
        """Test apply_trigger method for ASR evaluation."""
        data, _ = sample_data
        strategy = get_poisoning_strategy("badnets")
        
        # Apply trigger to single image
        image = data[0]
        triggered = strategy.apply_trigger(
            image,
            trigger_size=3,
            trigger_value=1.0,
            trigger_position="bottom_right",
        )
        
        assert triggered.shape == image.shape
        assert np.allclose(triggered[:, 29:, 29:], 1.0)
    
    def test_metadata_cache_key(self, sample_data):
        """Test that cache key is deterministic."""
        data, labels = sample_data
        strategy = get_poisoning_strategy("badnets")
        
        result1 = strategy.apply(
            data=data, labels=labels, dataset_id="test",
            poison_rate=0.1, target_class=0, seed=42,
        )
        
        result2 = strategy.apply(
            data=data, labels=labels, dataset_id="test",
            poison_rate=0.1, target_class=0, seed=42,
        )
        
        # Same config should produce same cache key
        assert result1.metadata.compute_cache_key() == result2.metadata.compute_cache_key()
        
        # Different config should produce different cache key
        result3 = strategy.apply(
            data=data, labels=labels, dataset_id="test",
            poison_rate=0.2, target_class=0, seed=42,  # Different rate
        )
        assert result1.metadata.compute_cache_key() != result3.metadata.compute_cache_key()


class TestLabelFlipStrategy:
    """Test label-flip poisoning strategy."""
    
    def test_basic_label_flip(self, sample_data):
        """Test basic label flipping."""
        data, labels = sample_data
        strategy = get_poisoning_strategy("label_flip")
        
        result = strategy.apply(
            data=data,
            labels=labels,
            dataset_id="test",
            poison_rate=0.1,
            seed=42,
            flip_mode="random",
        )
        
        assert isinstance(result, PoisoningResult)
        assert result.metadata.poison_type == PoisonType.LABEL_FLIP
        assert result.metadata.trigger_info is None  # No trigger
    
    def test_data_unchanged(self, sample_data):
        """Test that data is not modified in label-flip attack."""
        data, labels = sample_data
        strategy = get_poisoning_strategy("label_flip")
        
        result = strategy.apply(
            data=data,
            labels=labels,
            dataset_id="test",
            poison_rate=0.1,
            seed=42,
        )
        
        # Data should be unchanged (just copied)
        assert np.allclose(result.data, data)
    
    def test_labels_flipped(self, sample_data):
        """Test that labels are actually flipped."""
        data, labels = sample_data
        strategy = get_poisoning_strategy("label_flip")
        
        result = strategy.apply(
            data=data,
            labels=labels,
            dataset_id="test",
            poison_rate=0.1,
            seed=42,
        )
        
        # Check some labels were flipped
        for idx in result.metadata.poisoned_indices:
            # Flipped label should be different from original
            assert result.labels[idx] != result.original_labels[idx]
    
    def test_targeted_label_flip(self, sample_data):
        """Test targeted label flip."""
        data, labels = sample_data
        strategy = get_poisoning_strategy("label_flip")
        
        result = strategy.apply(
            data=data,
            labels=labels,
            dataset_id="test",
            poison_rate=0.1,
            target_class=5,
            flip_mode="targeted",
            seed=42,
        )
        
        # All flipped labels should be target class
        for idx in result.metadata.poisoned_indices:
            assert result.labels[idx] == 5
    
    def test_no_trigger_method(self, sample_data):
        """Test that apply_trigger raises NotImplementedError."""
        strategy = get_poisoning_strategy("label_flip")
        
        with pytest.raises(NotImplementedError):
            strategy.apply_trigger(sample_data[0][0])


class TestBlendStrategy:
    """Test Blend backdoor attack strategy."""
    
    def test_basic_blend(self, sample_data):
        """Test basic blend poisoning."""
        data, labels = sample_data
        strategy = get_poisoning_strategy("blend")
        
        result = strategy.apply(
            data=data,
            labels=labels,
            dataset_id="test",
            poison_rate=0.1,
            target_class=0,
            alpha=0.2,
            pattern="random",
            seed=42,
        )
        
        assert isinstance(result, PoisoningResult)
        assert result.metadata.poison_type == PoisonType.BACKDOOR
        assert result.metadata.trigger_info is not None
    
    def test_alpha_blending(self, sample_data):
        """Test that alpha blending is applied correctly."""
        data, labels = sample_data
        strategy = get_poisoning_strategy("blend")
        
        result = strategy.apply(
            data=data,
            labels=labels,
            dataset_id="test",
            poison_rate=0.1,
            target_class=0,
            alpha=0.5,
            pattern="checkerboard",
            seed=42,
        )
        
        # Poisoned samples should be modified (blended)
        for idx in result.metadata.poisoned_indices:
            # With alpha=0.5, output should be mix of original and pattern
            # Just check it's different from original
            assert not np.allclose(result.data[idx], data[idx])


class TestWaNetStrategy:
    """Test WaNet warping-based attack strategy."""
    
    def test_basic_wanet(self, sample_data):
        """Test basic WaNet poisoning."""
        data, labels = sample_data
        strategy = get_poisoning_strategy("wanet")
        
        result = strategy.apply(
            data=data,
            labels=labels,
            dataset_id="test",
            poison_rate=0.1,
            target_class=0,
            warp_strength=0.5,
            grid_size=4,
            seed=42,
        )
        
        assert isinstance(result, PoisoningResult)
        assert result.metadata.poison_type == PoisonType.BACKDOOR


class TestValidation:
    """Test parameter validation."""
    
    def test_invalid_poison_rate(self, sample_data):
        """Test validation of poison rate."""
        data, labels = sample_data
        strategy = get_poisoning_strategy("badnets")
        
        with pytest.raises(ValueError):
            strategy.apply(
                data=data,
                labels=labels,
                dataset_id="test",
                poison_rate=1.5,  # Invalid: > 1
                target_class=0,
            )
    
    def test_data_labels_mismatch(self):
        """Test validation of data/labels length."""
        data = np.random.rand(100, 3, 32, 32)
        labels = np.random.randint(0, 10, 50)  # Mismatched length
        
        strategy = get_poisoning_strategy("badnets")
        
        with pytest.raises(ValueError):
            strategy.apply(
                data=data,
                labels=labels,
                dataset_id="test",
                poison_rate=0.1,
                target_class=0,
            )


class TestPoisoningMetadata:
    """Test PoisoningMetadata class."""
    
    def test_metadata_serialization(self):
        """Test metadata can be serialized to dict and back."""
        metadata = PoisoningMetadata(
            technique_name="badnets",
            original_dataset_id="cifar10",
            poison_type=PoisonType.BACKDOOR,
            attack_goal=AttackGoal.TARGETED,
            poison_rate_requested=0.1,
            poison_rate_actual=0.098,
            num_poisoned=98,
            target_class=0,
            poisoned_indices=(1, 5, 10, 15),
            config={"trigger_size": 3},
            trigger_info={"technique": "badnets"},
            seed=42,
        )
        
        # Convert to dict
        d = metadata.to_dict()
        assert isinstance(d, dict)
        assert d["technique_name"] == "badnets"
        
        # Convert back
        metadata2 = PoisoningMetadata.from_dict(d)
        assert metadata2.technique_name == metadata.technique_name
        assert metadata2.num_poisoned == metadata.num_poisoned
    
    def test_metadata_immutable(self):
        """Test that metadata is immutable (frozen=True)."""
        metadata = PoisoningMetadata(
            technique_name="badnets",
            original_dataset_id="test",
            poison_type=PoisonType.BACKDOOR,
            attack_goal=AttackGoal.TARGETED,
            poison_rate_requested=0.1,
            poison_rate_actual=0.1,
            num_poisoned=10,
        )
        
        # Should raise error when trying to modify
        with pytest.raises(Exception):  # FrozenInstanceError
            metadata.num_poisoned = 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
