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

def prepare_backdoor_dataset(clean_dataset_dir, poisoned_dataset_dir) -> str:
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
    poison_frac = 0.05
    trigger_value = 1
    trigger_size = 3
    target_class = 0
    logger.info(f"Injecting backdoor samples targeting class {target_class}...")
    n_poison = int(len(image) * poison_frac)
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

    logger.info(f"Backdoor dataset prepared at {poisoned_dataset_dir}")