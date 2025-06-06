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
from typing import Dict, Optional, Tuple, Any, List
from pathlib import Path

from landseer_pipeline.evaluator.fingerprinting import evaluate_fingerprinting_mingd
from landseer_pipeline.config import Stage
from landseer_pipeline.utils import load_config_from_script

logger = logging.getLogger()

class ModelEvaluator:

    def __init__(self, settings, dataset_manager, attacks, outputs, gpu_id):
        self.dataset_manager = dataset_manager
        self.attacks = attacks
        self.model_config = settings.config.pipeline[Stage.DURING_TRAINING].noop.docker.config_script
        self.outputs = outputs
        self.device = f"cuda" if torch.cuda.is_available() else "cpu"

    def _get_model_path(self, model_name: str) -> str:
        for stage, outputs in self.outputs.items():
            if stage == Stage.POST_TRAINING:
                for output in outputs:
                    if isinstance(output, str):
                        output = Path(output)
                    if output.is_dir():
                        model_path = Path(output / model_name)
                        if model_path.exists():
                            return str(model_path)
        for stage, outputs in self.outputs.items():
            if stage == Stage.DURING_TRAINING:
                for output in outputs:
                    if isinstance(output, str):
                        output = Path(output)
                    if output.is_dir():
                        model_path = Path(output / model_name)
                        if model_path.exists():
                            return str(model_path)
        logger.warning(f"Model {model_name} not found in outputs.")

    
    def evaluate_model(self, dataset_path: str) -> Dict[str, float]:
        print(self.device.count)
        
        metrics = {}
        model_path = self._get_model_path("model.pt")
        logger.info(f"Evaluating model {model_path} on dataset {dataset_path}")
        if model_path.endswith('.pt'):
            metrics = self._evaluate_pytorch_model(model_path, dataset_path)
        elif model_path.endswith('.h5'):
            metrics = self._evaluate_tensorflow_model(model_path, dataset_path)
        else:
            logger.warning(f"Unsupported model format: {model_path}")
            metrics = {"clean_test_accuracy": 0.0}
        for name, value in metrics.items():
            print(f"Metric {name}")
            logger.info(f"Metric {name}: {value}")
                
        return metrics
    
    def _evaluate_pytorch_model(self, model_path: str, dataset_path: str):
        config = load_config_from_script(self.model_config)

        model = config().to(self.device)

        model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
        model.eval()
            
        clean_train_loader, clean_test_loader = self._load_dataset(dataset_path)
        print(f"dataset_path: {dataset_path}")
            
        metrics = {}
        logger.info(f"model path: {model_path}") 
        clean_train_acc = self.evaluate_clean(model, clean_train_loader)
        metrics["clean_train_accuracy"] = clean_train_acc
        clean_test_acc = self.evaluate_clean(model, clean_test_loader)
        metrics["clean_test_accuracy"] = clean_test_acc

        # outlier test
        ood_loader =  self.generate_ood_samples(dataset_path)
        if ood_loader is not None:
            try:
                ood_auc = self.evaluate_outlier(model, clean_test_loader, ood_loader)
                metrics["ood_auc"] = ood_auc
            except Exception as e:
                logger.warning(f"Could not evaluate outlier detection: {e}")
        else:
            metrics["ood_auc"] = 0.0
                        
        if clean_test_loader is not None:
            try:
                robust_acc = self.evaluate_pgd(model, clean_test_loader)
                metrics["robust_accuracy"] = robust_acc
            except Exception as e:
                logger.warning(f"Could not evaluate robust accuracy: {e}")
        else:
            metrics["robust_accuracy"] = 0.0
        if self.dataset_manager.poisoned_dataset_dir:
            poisoned_loader = None
            if os.path.exists(self.dataset_manager.poisoned_dataset_dir):
                poisoned_loader = self._load_dataset(self.dataset_manager.poisoned_dataset_dir)[1]
            if poisoned_loader is not None:
                try:
                    target_class = 0
                    backdoor_asr = self.evaluate_backdoor(model, poisoned_loader, target_class)
                    metrics["backdoor_asr"] = backdoor_asr
                except Exception as e:
                    logger.warning(f"Could not evaluate backdoor ASR: {e}")
        else:
            metrics["backdoor_asr"] = 0.0
        metrics["fingerprinting"] = evaluate_fingerprinting_mingd(model, clean_test_loader)
        metrics["privacy_epsilon"] = self.extract_privacy_epsilon(self.outputs)
             
        return metrics
    
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

    def _load_dataset(self, dataset_path: str) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
        clean_train_loader, clean_test_loader = None, None
             
        if dataset_path:
            files = os.listdir(dataset_path)
            if 'test_data.npy' in files and 'test_labels.npy' in files:
                X_train = np.load(os.path.join(dataset_path, 'data.npy'))
                y_train = np.load(os.path.join(dataset_path, 'labels.npy'))
                X_test = np.load(os.path.join(dataset_path, 'test_data.npy'))
                y_test = np.load(os.path.join(dataset_path, 'test_labels.npy'))
                clean_train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long())
                clean_train_loader = DataLoader(clean_train_dataset, batch_size=128, shuffle=False)
                clean_test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long())
                clean_test_loader = DataLoader(clean_test_dataset, batch_size=128 , shuffle=False)
        else:
            logger.warning(f"Unsupported dataset format: {dataset_path}")
        return clean_train_loader, clean_test_loader
    
    def evaluate_clean(self, model, loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                pred = model(X).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0.0

    def evaluate_pgd(self, model, loader):
        model.eval()
        atk = PGD(model, eps=8/255, alpha=2/255, steps=10)
        correct, total = 0, 0
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            adv_X = atk(X, y)
            with torch.no_grad():
                pred = model(adv_X).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0.0

    def evaluate_outlier(self, model, clean_loader, ood_loader):
        model.eval()
        scores, labels = [], []
        
        for X, _ in clean_loader:
            X = X.to(self.device)
            with torch.no_grad():
                out = torch.softmax(model(X), dim=1)
                max_conf = out.max(1)[0]
            scores.extend((1 - max_conf).cpu().numpy())
            labels.extend([0] * X.size(0))
        
        for X, _ in ood_loader:
            X = X.to(self.device)
            with torch.no_grad():
                out = torch.softmax(model(X), dim=1)
                max_conf = out.max(1)[0]
            scores.extend((1 - max_conf).cpu().numpy())
            labels.extend([1] * X.size(0))
        
        return roc_auc_score(labels, scores)

    def evaluate_backdoor(self, model, poisoned_loader, target_class):
        model.eval()
        total, target_hits = 0, 0
        with torch.no_grad():
            for X, _ in poisoned_loader:
                X = X.to(self.device)
                pred = model(X).argmax(1)
                target_hits += (pred == target_class).sum().item()
                total += X.size(0)
        return target_hits / total
    
    def generate_ood_samples(self, dataset_path: str):
        ood_loader = None
        if dataset_path:
            files = os.listdir(dataset_path)
            if 'test_data.npy' in files and 'test_labels.npy' in files:
                X_train = np.load(os.path.join(dataset_path, 'data.npy'))
                y_train = np.load(os.path.join(dataset_path, 'labels.npy'))
                X_test = np.load(os.path.join(dataset_path, 'test_data.npy'))
                y_test = np.load(os.path.join(dataset_path, 'test_labels.npy'))
                X_train = torch.tensor(X_train).float()
                y_train = torch.tensor(y_train).long()
                X_test = torch.tensor(X_test).float()
                y_test = torch.tensor(y_test).long()
                ood_loader = DataLoader(TensorDataset(torch.rand_like(X_test), y_test), batch_size=64, shuffle=False)
        return ood_loader
    
    def extract_privacy_epsilon(self, output_by_stages: Dict) -> float:
        for stage, outputs in output_by_stages.items():
            for output in outputs:
                if isinstance(output, str):
                    output = Path(output)
                if isinstance(output, Path) and output.is_dir():
                    params_file = output / "privacy_metrics.txt"
                    if params_file.exists():
                        epsilon = self._parse_privacy_params(params_file)
                        if epsilon >= 0:
                            return (epsilon, stage)
        logger.warning("No privacy metrics file found in outputs.")
        return -1.0
        
    def _parse_privacy_params(self, params_file: Path) -> float:
        try:
            with open(params_file, 'r') as f:
                for line in f:
                    if line.startswith("epsilon="):
                        return float(line.split('=')[1].strip())
        except Exception as e:
            logger.warning(f"Could not parse privacy parameters from {params_file}: {e}")
        return -1.0
