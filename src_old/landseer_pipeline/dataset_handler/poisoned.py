import logging
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

logger = logging.getLogger(__name__)

def add_backdoor_trigger(images, trigger_value=1.0, size=3):
    """
    Applies a square trigger of given size and value to the bottom-right corner of each image.
    images: [N, C, H, W]
    """
    images = images.clone()
    _, _, h, w = images.shape
    images[:, :, h - size:, w - size:] = trigger_value
    return images

def prepare_backdoor_dataset(clean_dataset_dir, poisoned_dataset_dir, poison_fraction=0.05, trigger_value=1.0, trigger_size=3, target_class=0):
    """
    Prepares a backdoor dataset by injecting triggers into a fraction of the training images.
    """
    logger.info(f"Preparing backdoor dataset ...")
    data_np = np.load( clean_dataset_dir / "data.npy")
    labels_np = np.load(clean_dataset_dir / "labels.npy")
    test_data_np = np.load(clean_dataset_dir / "test_data.npy")
    test_labels_np = np.load(clean_dataset_dir / "test_labels.npy")
    data = torch.tensor(data_np)
    labels = torch.tensor(labels_np)
    if len(data.shape) == 3:
        data = data.unsqueeze(1)
    dataset = TensorDataset(data, labels)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    image, label = next(iter(loader))
    poison_fraction = 0.05
    trigger_value = 1
    trigger_size = 3
    target_class = 0
    logger.info(f"Injecting backdoor samples targeting class {target_class}...")
    n_poison = int(len(image) * poison_fraction)
    indices = torch.randperm(len(image))[:n_poison]
    data_poisoned = add_backdoor_trigger(
        image[indices], trigger_value, trigger_size)
    labels_poisoned = torch.full(
        (n_poison,), target_class, dtype=torch.long)
    
    images = torch.cat((image, data_poisoned), dim=0)
    labels = torch.cat((label, labels_poisoned), dim=0)
    np.save(poisoned_dataset_dir / "data.npy", images.cpu().numpy())
    np.save(poisoned_dataset_dir / "labels.npy", labels.cpu().numpy())
    np.save(poisoned_dataset_dir / "test_data.npy", test_data_np)
    np.save(poisoned_dataset_dir / "test_labels.npy", test_labels_np)

    # Create metadata file for masking defense compatibility
    create_backdoor_metadata(poisoned_dataset_dir, poison_fraction, target_class, trigger_size, trigger_value)

    logger.info(f"Backdoor dataset prepared at {poisoned_dataset_dir}")


def create_backdoor_metadata(output_dir, poison_fraction, target_class, trigger_size, trigger_value):
    """Create metadata file for backdoor attack compatibility with masking defense."""
    import json
    
    # Create transition matrix for backdoor attack
    transition_matrix = []
    for i in range(10):
        row = [0.0] * 10
        if i == target_class:
            row[i] = 1.0  # Target class samples stay as target class
        else:
            row[i] = 1.0 - poison_fraction  # Clean samples
            row[target_class] = poison_fraction  # Poisoned samples go to target
        transition_matrix.append(row)
    
    metadata = {
        "poison_type": "backdoor",
        "noise_rate": poison_fraction,
        "description": f"Backdoor attack with {trigger_size}x{trigger_size} trigger targeting class {target_class}",
        "attack_info": {
            "method": "backdoor_trigger",
            "trigger_size": trigger_size,
            "trigger_value": trigger_value,
            "trigger_position": "bottom_right",
            "target_class": target_class,
            "poison_fraction": poison_fraction
        },
        "transition_matrix": transition_matrix,
        "classes": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    }
    
    # Save metadata
    metadata_file = output_dir / 'poisoning_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Created backdoor metadata at: {metadata_file}")