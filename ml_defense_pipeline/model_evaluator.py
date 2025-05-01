"""
Model evaluation for ML Defense Pipeline
"""
import logging
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchattacks import PGD
from sklearn.metrics import roc_auc_score
from typing import Optional

logger = logging.getLogger("defense_pipeline")

class ModelEvaluator:
    """Evaluates models and computes metrics"""
    
    def evaluate_model(self, model_path: str, dataset_path: str) -> float:
        """
        Evaluate a model on a dataset
        
        Args:
            model_path: Path to the model in .h5 format
            dataset_path: Path to the dataset in .h5 format
            
        Returns:
            Accuracy as a float
        """
        logger.info(f"Evaluating model {model_path} on dataset {dataset_path}")
        
        try:
            import h5py
            import tensorflow as tf
            import numpy as np
            
            # Load test data from dataset
            with h5py.File(dataset_path, 'r') as ds:
                if "X_test" in ds and "y_test" in ds:
                    X_test = ds["X_test"][:]
                    y_test = ds["y_test"][:]
                else:
                    logger.warning("Dataset doesn't contain test data")
                    return 0.0
            
            # Load and evaluate model
            try:
                model = tf.keras.models.load_model(model_path)
                results = model.evaluate(X_test, y_test, verbose=0)
                
                if isinstance(results, list):
                    acc = results[1] if len(results) > 1 else results[0]
                else:
                    acc = results
                    
                logger.info(f"Model accuracy: {acc:.4f}")
                return acc
                
            except Exception as e:
                logger.warning(f"Could not evaluate model using TensorFlow: {e}")
                return 0.0
                
        except ImportError as e:
            logger.warning(f"Could not evaluate model: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return 0.0
    
    def train_baseline_model(self, dataset_path: str, output_path: str, device) -> float:
        """
        Train a baseline model for comparison
        
        Args:
            dataset_path: Path to dataset in .h5 format
            output_path: Path to save the baseline model
            
        Returns:
            Baseline accuracy as a float
        """
        logger.info("Training baseline model for comparison...")
        
        # train baseline model using the config_model.py script
		# TODO: implement the training logic here 
        
		#default path right now is config_model.py load config from there

        from config_model import config
        model = config().to(device)
        model.load_state_dict(torch.load(output_path, map_location=device), strict=False)
def evaluate_clean(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def evaluate_pgd(model, loader, device):
    model.eval()
    atk = PGD(model, eps=8/255, alpha=2/255, steps=10)
    correct, total = 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        adv_X = atk(X, y)
        with torch.no_grad():
            pred = model(adv_X).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def evaluate_outlier(model, clean_loader, ood_loader, device):
    model.eval()
    scores, labels = [], []
    for loader, label in [(clean_loader, 0), (ood_loader, 1)]:
        for X, _ in loader:
            X = X.to(device)
            with torch.no_grad():
                out = torch.softmax(model(X), dim=1)
                max_conf = out.max(1)[0]
            scores.extend(1 - max_conf.cpu().numpy())
            labels.extend([label] * X.size(0))
    return roc_auc_score(labels, scores)


def evaluate_backdoor(model, poisoned_loader, target_class, device):
    model.eval()
    total, target_hits = 0, 0
    with torch.no_grad():
        for X, _ in poisoned_loader:
            X = X.to(device)
            pred = model(X).argmax(1)
            target_hits += (pred == target_class).sum().item()
            total += X.size(0)
    return target_hits / total