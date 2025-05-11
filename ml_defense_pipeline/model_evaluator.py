"""
Model evaluation for ML Defense Pipeline
"""
import logging
import os
import torch
import numpy as np
import h5py
from torch.utils.data import TensorDataset, DataLoader
from torchattacks import PGD
from sklearn.metrics import roc_auc_score
from typing import Dict, Optional, Tuple, Any

logger = logging.getLogger("defense_pipeline")

class ModelEvaluator:
    
    def evaluate_model(self, model_path: str, dataset_path: str, device=None) -> Dict[str, float]:
        logger.info(f"Evaluating model {model_path} on dataset {dataset_path}")
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        metrics = {}
        
        try:
            # Determine model type based on file extension
            if model_path.endswith('.pt'):
                metrics = self._evaluate_pytorch_model(model_path, dataset_path, device)
            elif model_path.endswith('.h5'):
                metrics = self._evaluate_tensorflow_model(model_path, dataset_path)
            else:
                logger.warning(f"Unsupported model format: {model_path}")
                metrics = {"clean_test_accuracy": 0.0}
            
            # Log metrics
            for name, value in metrics.items():
                logger.info(f"Metric {name}: {value:.4f}")
                
            return metrics
                
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {"clean_test_accuracy": 0.0}
    
    def _evaluate_pytorch_model(self, model_path: str, dataset_path: str, device) -> Dict[str, float]:
        try:
            from config_model import config
            
            model = config().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
            model.eval()
            
            clean_loader, poisoned_loader, ood_loader = self._load_dataset(dataset_path, device)
            
            metrics = {}
            
            clean_acc = self.evaluate_clean(model, clean_loader, device)
            metrics["clean_test_accuracy"] = clean_acc
            
            if clean_loader is not None:
                try:
                    robust_acc = self.evaluate_pgd(model, clean_loader, device)
                    metrics["robust_accuracy"] = robust_acc
                except Exception as e:
                    logger.warning(f"Could not evaluate robust accuracy: {e}")
            
            if ood_loader is not None:
                try:
                    ood_auc = self.evaluate_outlier(model, clean_loader, ood_loader, device)
                    metrics["ood_auc"] = ood_auc
                except Exception as e:
                    logger.warning(f"Could not evaluate outlier detection: {e}")
            
            if poisoned_loader is not None:
                try:
                    target_class = 0
                    backdoor_asr = self.evaluate_backdoor(model, poisoned_loader, target_class, device)
                    metrics["backdoor_asr"] = backdoor_asr
                except Exception as e:
                    logger.warning(f"Could not evaluate backdoor ASR: {e}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to evaluate PyTorch model: {e}")
            return {"clean_test_accuracy": 0.0}
    
    def _evaluate_tensorflow_model(self, model_path: str, dataset_path: str) -> Dict[str, float]:
        try:
            import tensorflow as tf
            with h5py.File(dataset_path, 'r') as ds:
                if "X_test" in ds and "y_test" in ds:
                    X_test = ds["X_test"][:]
                    y_test = ds["y_test"][:]
                else:
                    logger.warning("Dataset doesn't contain test data")
                    return {"clean_test_accuracy": 0.0}
            
            # Load and evaluate model
            try:
                model = tf.keras.models.load_model(model_path)
                results = model.evaluate(X_test, y_test, verbose=0)
                
                if isinstance(results, list):
                    acc = results[1] if len(results) > 1 else results[0]
                else:
                    acc = results
                    
                return {"clean_test_accuracy": float(acc)}
                
            except Exception as e:
                logger.warning(f"Could not evaluate model using TensorFlow: {e}")
                return {"clean_test_accuracy": 0.0}
                
        except ImportError as e:
            logger.warning(f"Could not evaluate TensorFlow model: {e}")
            return {"clean_test_accuracy": 0.0}

    def _load_dataset(self, dataset_path: str, device) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
        """Load dataset and create dataloaders for clean, poisoned, and OOD data"""
        clean_loader, poisoned_loader, ood_loader = None, None, None
        
        try:
        
            if dataset_path.endswith('.npy'):
                files = os.listdir(dataset_path)
                if 'X_test.npy' in files and 'y_test.npy' in files:
                    X_test = np.load(os.path.join(dataset_path, 'X_test.npy'))
                    y_test = np.load(os.path.join(dataset_path, 'y_test.npy'))
                    clean_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long())
                    clean_loader = DataLoader(clean_dataset, batch_size=128, shuffle=False)
                if 'X_poisoned.npy' in files and 'y_poisoned.npy' in files:
                    X_poisoned = np.load(os.path.join(dataset_path, 'X_poisoned.npy'))
                    y_poisoned = np.load(os.path.join(dataset_path, 'y_poisoned.npy'))
                    poisoned_dataset = TensorDataset(torch.tensor(X_poisoned).float(), torch.tensor(y_poisoned).long())
                    poisoned_loader = DataLoader(poisoned_dataset, batch_size=128, shuffle=False)
                if 'X_ood.npy' in files and 'y_ood.npy' in files:
                    X_ood = np.load(os.path.join(dataset_path, 'X_ood.npy'))
                    y_ood = np.load(os.path.join(dataset_path, 'y_ood.npy'))
                    ood_dataset = TensorDataset(torch.tensor(X_ood).float(), torch.tensor(y_ood).long())
                    ood_loader = DataLoader(ood_dataset, batch_size=128, shuffle=False)
            else:
                logger.warning(f"Unsupported dataset format: {dataset_path}")
        
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
        
        return clean_loader, poisoned_loader, ood_loader
    
    def train_baseline_model(self, dataset_path: str, output_path: str, device) -> Dict[str, float]:
        logger.info("Training baseline model for comparison...")
        try:
            if os.path.exists(output_path):
                logger.info(f"Baseline model already exists at {output_path}")
                return self.evaluate_model(output_path, dataset_path, device)
            
            clean_loader, _, _ = self._load_dataset(dataset_path, device)
            
            if clean_loader is None:
                logger.error("Could not load dataset for baseline training")
                return {"clean_test_accuracy": 0.0}
                
            from config_model import config
            model = config().to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()
            
            model.train()
            for epoch in range(10):  # Minimal training
                total_loss = 0
                for X, y in clean_loader:
                    X, y = X.to(device), y.to(device)
                    optimizer.zero_grad()
                    outputs = model(X)
                    loss = criterion(outputs, y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                logger.info(f"Epoch {epoch+1}/10, Loss: {total_loss/len(clean_loader):.4f}")
            
            # Save model
            torch.save(model.state_dict(), output_path)
            logger.info(f"Baseline model saved to {output_path}")
            
            # Evaluate the trained model
            model.eval()
            return self.evaluate_model(output_path, dataset_path, device)
            
        except Exception as e:
            logger.error(f"Error training baseline model: {e}")
            return {"clean_test_accuracy": 0.0}
    
    def evaluate_clean(self, model, loader, device):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                pred = model(X).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0.0

    def evaluate_pgd(self, model, loader, device):
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
        return correct / total if total > 0 else 0.0

    def evaluate_outlier(self, model, clean_loader, ood_loader, device):
        model.eval()
        scores, labels = [], []
        
        for X, _ in clean_loader:
            X = X.to(device)
            with torch.no_grad():
                out = torch.softmax(model(X), dim=1)
                max_conf = out.max(1)[0]
            scores.extend((1 - max_conf).cpu().numpy())
            labels.extend([0] * X.size(0))
        
        for X, _ in ood_loader:
            X = X.to(device)
            with torch.no_grad():
                out = torch.softmax(model(X), dim=1)
                max_conf = out.max(1)[0]
            scores.extend((1 - max_conf).cpu().numpy())
            labels.extend([1] * X.size(0))
        
        return roc_auc_score(labels, scores)

    def evaluate_backdoor(self, model, poisoned_loader, target_class, device):
        model.eval()
        total, target_hits = 0, 0
        with torch.no_grad():
            for X, _ in poisoned_loader:
                X = X.to(device)
                pred = model(X).argmax(1)
                target_hits += (pred == target_class).sum().item()
                total += X.size(0)			

        return target_hits / total if total > 0 else 0.0
