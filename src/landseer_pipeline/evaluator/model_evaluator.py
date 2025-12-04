"""
Model evaluation for ML Defense Pipeline
"""
import logging
import os
import json
import torch
import numpy as np
import h5py
from torch.utils.data import TensorDataset, DataLoader
from torchattacks import PGD
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Dict, Optional, Tuple, Any, List
from pathlib import Path

from landseer_pipeline.evaluator.fingerprinting import evaluate_fingerprinting_mingd
from landseer_pipeline.config import Stage
from landseer_pipeline.utils import load_config_from_script
from ..utils.onnx_converter import get_model_framework
from ..utils.model_format_manager import get_model_format_manager

import torch.nn.functional as F

import json
from torchvision import datasets, transforms
import os

logger = logging.getLogger()

#Adding dataset Normalizer
# class NormalizedDataset(torch.utils.data.Dataset):
#     def __init__(self, images, labels, normalize=True):
#         images = torch.tensor(images).float()
#         if images.max() > 1.0:
#             images = images / 255.0
#         self.images = images
#         self.labels = torch.tensor(labels).long()
#         self.transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if normalize else None

#     def __getitem__(self, idx):
#         x = self.images[idx]
#         if self.transform:
#             x = self.transform(x)
#         y = self.labels[idx]
#         return x, y

#     def __len__(self):
#         return len(self.labels)

class ModelEvaluator:

    def __init__(self, settings, dataset_manager, attacks, tools_by_stage, gpu_id, combination_id, combination_output):
        self.dataset_manager = dataset_manager
        self.attacks = attacks
        # Use centralized model script (fallback to deprecated per-tool noop config_script if absent)
        self.model_script_path = None
        self.format_manager = get_model_format_manager()

        try:
            if getattr(settings.config, 'model', None):
                self.model_script_path = getattr(settings.config.model, 'script', None)
        except Exception:
            pass
        if not self.model_script_path:
            logger.error("ModelEvaluator: No model script found (central or noop). Model reconstruction may fail.")
        self.tools_by_stage = tools_by_stage
        #print(f"ModelEvaluator: CUDA devices available - {torch.cuda.device_count()}")
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id is not None else "cpu"
        self.combination_id = combination_id
        self.combination_id = combination_id
        self.combination_output = combination_output

    #I also need something to see which types of tools are in pipeline
    @property
    def input_model(self) -> str:
        json_path = Path(self.combination_output) / "fin_output_paths.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                paths = json.load(f)
            model_entry = paths.get("model.onnx", "")
            if isinstance(model_entry, dict):
                return model_entry.get("source_path", "")
            return model_entry
        else:
            logger.warning(f"{self.combination_id}: Combination output paths file {json_path} does not exist.")
            return ""

    @property
    def input_train_dataset(self) -> str:
        json_path = Path(self.combination_output) / "fin_output_paths.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                paths = json.load(f)
            data_entry = paths.get("data.npy", "")
            if isinstance(data_entry, dict):
                return data_entry.get("source_path", "")
            return data_entry
        else:
            logger.warning(f"{self.combination_id}: Combination output paths file {json_path} does not exist.")
            return ""

    @property
    def input_test_dataset(self) -> str:
        json_path = Path(self.combination_output) / "fin_output_paths.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                paths = json.load(f)
            test_data_entry = paths.get("test_data.npy", "")
            if isinstance(test_data_entry, dict):
                return test_data_entry.get("source_path", "")
            return test_data_entry
        else:
            logger.warning(f"{self.combination_id}: Combination output paths file {json_path} does not exist.")
            return ""

    @property
    def input_train_labels(self) -> str:
        json_path = Path(self.combination_output) / "fin_output_paths.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                paths = json.load(f)
            labels_entry = paths.get("labels.npy", "")
            if isinstance(labels_entry, dict):
                return labels_entry.get("source_path", "")
            return labels_entry
        else:
            logger.warning(f"{self.combination_id}:Combination output paths file {json_path} does not exist.")
            return ""

    @property
    def input_test_labels(self) -> str:
        json_path = Path(self.combination_output) / "fin_output_paths.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                paths = json.load(f)
            test_labels_entry = paths.get("test_labels.npy", "")
            if isinstance(test_labels_entry, dict):
                return test_labels_entry.get("source_path", "")
            return test_labels_entry
        else:
            logger.warning(f"{self.combination_id}: Combination output paths file {json_path} does not exist.")
            return ""

    @property
    def input_wm_matrix(self) -> str:
        json_path = Path(self.combination_output) / "fin_output_paths.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                paths = json.load(f)
            wm_matrix_entry = paths.get("wm_matrix.npy", "")
            if isinstance(wm_matrix_entry, dict):
                return wm_matrix_entry.get("source_path", "")
            return wm_matrix_entry
        else:
            logger.warning(f"{self.combination_id}: Combination output paths file {json_path} does not exist.")
            return ""

    @property
    def input_wm_bits(self) -> str:
        json_path = Path(self.combination_output) / "fin_output_paths.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                paths = json.load(f)
            wm_bits_entry = paths.get("wm_bits.npy", "")
            if isinstance(wm_bits_entry, dict):
                return wm_bits_entry.get("source_path", "")
            return wm_bits_entry
        else:
            logger.warning(f"{self.combination_id}: Combination output paths file {json_path} does not exist.")
            return ""

    def extract_defense_types_from_tools(self, tools_by_stage) -> List[str]:
        """Extract defense types from tool configurations based on Docker image labels"""
        defense_types = set()
        
        for stage, tools in tools_by_stage.items():
            for tool in tools:
                try:
                    # Handle both tool objects and string names
                    tool_name = tool.name if hasattr(tool, 'name') else str(tool)
                    
                    # Get defense type from Docker image labels
                    if hasattr(tool, 'tool_defense_type'):
                        defense_type = tool.tool_defense_type
                        if defense_type and defense_type != "unknown":
                            defense_types.add(defense_type)
                            logger.info(f"{self.combination_id}: Tool '{tool_name}' has defense_type: {defense_type}")
                        else:
                            logger.warning(f"{self.combination_id}: Tool '{tool_name}' has no defense_type or unknown defense_type")
                    else:
                        logger.warning(f"{self.combination_id}: Tool '{tool_name}' does not have tool_defense_type property")
                except Exception as e:
                    logger.warning(f"{self.combination_id}: Error extracting defense type from tool '{tool}': {e}")

        return list(defense_types)

    def determine_applicable_attacks(self, defense_types: List[str]) -> List[str]:
        """Map defense types to applicable attack types for evaluation"""
        attack_types = set()
        
        # Always include clean evaluation
        attack_types.add("clean")
        attack_types.add("adversarial")
        attack_types.add("carlini")
        
        for defense_type in defense_types:
            defense_type_lower = defense_type.lower()
            
            if defense_type_lower == "adversarial":
                attack_types.add("adversarial")
                attack_types.add("pgd")
            elif defense_type_lower == "outlier_removal":
                attack_types.add("outlier")
                attack_types.add("ood")
            elif defense_type_lower == "differential_privacy":
                attack_types.add("privacy")
            elif defense_type_lower == "watermarking":
                attack_types.add("watermark")
            elif defense_type_lower == "fingerprinting":
                attack_types.add("fingerprinting")
            elif defense_type_lower == "backdoor":
                attack_types.add("backdoor")
            elif defense_type_lower == "carlini":
                attack_types.add("carlini")
            else:
                logger.warning(f"{self.combination_id}: Unknown defense type: {defense_type}, adding general evaluation")
                attack_types.add("general")
        
        # If no specific defense types found, run comprehensive evaluation
        if len(attack_types) == 1:  # Only "clean"
            attack_types.update(["adversarial", "outlier", "backdoor", "fingerprinting"])
            logger.info(f"{self.combination_id}: No specific defense types detected, running comprehensive evaluation")

        return list(attack_types)

    def get_dataset_directory_from_paths(self) -> str:
        """Get the directory containing the dataset files from the paths"""
        train_data_path = self.input_train_dataset
        if train_data_path and train_data_path != "":
            return str(Path(train_data_path).parent)
        
        test_data_path = self.input_test_dataset
        if test_data_path and test_data_path != "":
            return str(Path(test_data_path).parent)
        
        logger.warning("No dataset paths found in fin_output_paths.json")
        return ""


    def _get_model_path(self, model_name: str = "model.pt") -> str:
        """Get model path from fin_output_paths.json"""
        return self.input_model

    def evaluate_model(self, dataset_path: str) -> Dict[str, float]:
        """Evaluate model using attacks determined by tool defense types"""
        logger.info(f"{self.combination_id}: Starting model evaluation...")
        
        # Extract defense types from tools
        defense_types = self.extract_defense_types_from_tools(self.tools_by_stage)
        logger.info(f"{self.combination_id}: Detected defense types: {defense_types}")
        
        # Determine applicable attacks based on defense types
        applicable_attacks = self.determine_applicable_attacks(defense_types)
        logger.info(f"{self.combination_id}: Running attacks: {applicable_attacks}")

        # Get dataset directory from combination output paths
        dataset_dir = self.get_dataset_directory_from_paths()
        if not dataset_dir:
            logger.warning(f"{self.combination_id}: Using provided dataset_path as fallback")
            dataset_dir = dataset_path
        
        # Get model path from combination output
        model_path = self._get_model_path()
        if not model_path or model_path == "":
            logger.error(f"{self.combination_id}: No model path found for evaluation")
            return {"clean_test_accuracy": 0.0}
        elif not os.path.exists(model_path):
            logger.error(f"{self.combination_id}: Model path does not exist: {model_path}")
            return {"clean_test_accuracy": 0.0}
        elif not (isinstance(model_path, str) and (model_path.endswith('.onnx'))):
            model_path_converted = self._convert_to_pytorch_for_evaluation(model_path, source_format="onnx")
            if model_path_converted:
                model_path = model_path_converted
                logger.info(f"{self.combination_id}: Converted model to PyTorch for evaluation: {model_path}")
            else:
                logger.error(f"{self.combination_id}: Failed to convert model to PyTorch for evaluation")
                return {"clean_test_accuracy": 0.0}

        logger.info(f"{self.combination_id}: Evaluating model {model_path} on dataset {dataset_dir}")
        
        metrics = {}
        
        if isinstance(model_path, str) and model_path.endswith('.pt'):
            metrics = self._evaluate_pytorch_model(applicable_attacks)
        else:
            logger.warning(f"{self.combination_id}: Unsupported model format: {model_path}")
            metrics = {"clean_test_accuracy": 0.0}
        
        for name, value in metrics.items():
            logger.info(f"{self.combination_id}: Metric {name}: {value}")

        return metrics

    def _convert_to_pytorch_for_evaluation(self, model_path: str, source_format: str) -> Optional[str]:
        """Convert model to PyTorch for evaluation"""
        try:
            temp_pytorch_path = os.path.join(
                self.format_manager.temp_dir,
                f"eval_model_{self.combination_id}.pt"
            )
            
            success, _ = self.format_manager.converter.convert_model(
                model_path, temp_pytorch_path,
                source_framework=source_format,
                target_framework="pytorch",
                model_script_path=self.model_script_path
            )
            
            return temp_pytorch_path if success else None
            
        except Exception as e:
            logger.error(f"Failed to convert {source_format} model to PyTorch: {e}")
            return None
    
    def _evaluate_pytorch_model(self, applicable_attacks: List[str]):
        """Evaluate PyTorch model with attacks based on defense types"""
        # Get model and dataset paths from class properties
        model_path = self.input_model
        dataset_path = self.get_dataset_directory_from_paths()

        logger.info(f"{self.combination_id}: Model path: {model_path} (type: {type(model_path)})")
        logger.info(f"{self.combination_id}: Dataset path: {dataset_path} (type: {type(dataset_path)})")

        if not dataset_path:
            logger.error(f"{self.combination_id}: No valid dataset path found, cannot proceed with evaluation")
            return {"clean_test_accuracy": 0.0}
        
        if not self.model_script_path:
            logger.error(f"{self.combination_id}: No model script available to construct model architecture.")
            return {"clean_test_accuracy": 0.0}
        try:
            config = load_config_from_script(self.model_script_path)
            model = config().to(self.device)
        except Exception as e:
            logger.error(f"{self.combination_id}: Failed to construct model from script {self.model_script_path}: {e}")
            return {"clean_test_accuracy": 0.0}
        
        # Load model state dict with proper device mapping
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            logger.debug(f"{self.combination_id}: Direct device loading failed for {model_path}: {e}")
            # Try loading with CPU mapping first, then move to device
            try:
                state_dict = torch.load(model_path, map_location='cpu')
                model.load_state_dict(state_dict, strict=False)
                model = model.to(self.device)
                logger.info(f"{self.combination_id}: Successfully loaded model with CPU mapping and moved to {self.device}")
            except Exception as e2:
                logger.error(f"{self.combination_id}: Failed to load model even with CPU mapping: {e2}")
                return {"clean_test_accuracy": 0.0}
        
        model.eval()
            
        clean_train_loader, clean_test_loader = self._load_dataset(dataset_path)
        logger.info(f"{self.combination_id}: Loading dataset from: {dataset_path}")

        metrics = {}
        logger.info(f"{self.combination_id}: Evaluating model: {model_path}")
        
        # 1. Clean train and test accuracy (always performed)
        if clean_train_loader is not None:
            clean_train_acc = self.evaluate_clean(model, clean_train_loader)
            metrics["clean_train_accuracy"] = clean_train_acc
        else:
            metrics["clean_train_accuracy"] = 0.0
            
        if clean_test_loader is not None:
            clean_test_acc = self.evaluate_clean(model, clean_test_loader)
            metrics["clean_test_accuracy"] = clean_test_acc
            robust_acc = self.evaluate_pgd(model, clean_test_loader)
            metrics["pgd_acc"] = robust_acc
            carlini_acc = self.evaluate_carlini_l2(model, clean_test_loader, confidence=0)
            metrics["carlini_l2_accuracy"] = carlini_acc
            ood_loader = self.generate_ood_samples("cifar100")
            ood_auc = self.evaluate_outlier(model, clean_test_loader, ood_loader)
            metrics["ood_auc"] = ood_auc
            metrics["mia_auc"], metrics["eps_estimate"] = self.margin_auc_from_loaders(model, clean_train_loader, clean_test_loader, delta_dp=1e-5, device=self.device)
            fingerprint_score = evaluate_fingerprinting_mingd(model, clean_test_loader)
            metrics["fingerprinting"] = fingerprint_score
            logger.info(f"{self.combination_id}: Fingerprinting evaluation completed: {fingerprint_score}")

            if hasattr(self, 'dataset_manager') and self.dataset_manager.poisoned_dataset_dir:
                poisoned_loader = None
                if os.path.exists(self.dataset_manager.poisoned_dataset_dir):
                    poisoned_loader = self._load_dataset(self.dataset_manager.poisoned_dataset_dir)[1]
                    if poisoned_loader is not None:
                        target_class = 0
                        backdoor_asr = self.evaluate_backdoor(model, poisoned_loader, target_class)
                        metrics["backdoor_asr"] = backdoor_asr
                        logger.info(f"{self.combination_id}: Backdoor evaluation completed: {backdoor_asr}")
        else:
            metrics["clean_test_accuracy"] = -1.0
            metrics["pgd_acc"] = -1.0
            metrics["carlini_l2_accuracy"] = -1.0
            metrics["ood_auc"] = -1.0
            metrics["mia_auc"] = -1.0
            metrics["eps_estimate"] = -1.0
            metrics["backdoor_asr"] = -1.0
            metrics["fingerprinting"] = -1.0
        # Run fingerprinting if fingerprinting defense detected
        
        # Run watermarking evaluation if watermarking defense detected
        if "watermark" in applicable_attacks:
            try:
                watermark_accuracy = self.evaluate_watermark(model)
                metrics["watermark_accuracy"] = watermark_accuracy
                logger.info(f"{self.combination_id}: Watermarking evaluation completed: {watermark_accuracy}")
            except Exception as e:
                logger.warning(f"{self.combination_id}: Could not evaluate watermarking: {e}")
                metrics["watermark_accuracy"] = -1.0

        # Run privacy evaluation if differential privacy defense detected  
        if "privacy" in applicable_attacks:
            try:
                metrics["privacy_epsilon"] = self.extract_privacy_epsilon()
                metrics["dp_accuracy"] = self.extract_dp_accuracy()
                # Use clean test accuracy as dp_accuracy when DP is in pipeline
                #check if instage tool is dp
                #if "clean_test_accuracy" in metrics:
                #    metrics["clean_test_accuracy"] = metrics["dp_accuracy"]
                logger.info(f"{self.combination_id}: Privacy evaluation completed - dp_accuracy set to clean_test_accuracy: {metrics.get('dp_accuracy', -1.0)}")
            except Exception as e:
                logger.warning(f"{self.combination_id}: Could not evaluate privacy metrics: {e}")
                metrics["privacy_epsilon"] = -1.0
                metrics["dp_accuracy"] = -1.0

        return metrics
    
    """ def _evaluate_tensorflow_model(self, model_path: str, dataset_path: str) -> Dict[str, float]:
        try:
            import tensorflow as tf
            with h5py.File(dataset_path, 'r') as ds:
                if "X_test" in ds and "y_test" in ds:
                    X_test = ds["X_test"][:]
                    y_test = ds["y_test"][:]
                else:
                    logger.warning(f"{self.combination_id}: Dataset doesn't contain test data")
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
                logger.warning(f"{self.combination_id}: Could not evaluate model using TensorFlow: {e}")
                return {"clean_test_accuracy": 0.0}
                
        except ImportError as e:
            logger.warning(f"{self.combination_id}: Could not evaluate TensorFlow model: {e}")
            return {"clean_test_accuracy": 0.0}
            """
        
    # def _load_dataset(self, dataset_path: str) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    #     clean_train_loader, clean_test_loader = None, None

    #     if dataset_path:
    #         if not isinstance(dataset_path, (str, os.PathLike)):
    #             logger.error(f"dataset_path is not a string or PathLike object: {type(dataset_path)} = {dataset_path}")
    #             return clean_train_loader, clean_test_loader
            
    #         try:
    #             files = os.listdir(dataset_path)
    #         except Exception as e:
    #             logger.error(f"Failed to list directory {dataset_path}: {e}")
    #             return clean_train_loader, clean_test_loader
                
    #         if 'test_data.npy' in files and 'test_labels.npy' in files:
    #             X_train = np.load(os.path.join(dataset_path, 'data.npy'))
    #             y_train = np.load(os.path.join(dataset_path, 'labels.npy'))
    #             X_test = np.load(os.path.join(dataset_path, 'test_data.npy'))
    #             y_test = np.load(os.path.join(dataset_path, 'test_labels.npy'))

    #             train_dataset = NormalizedDataset(X_train, y_train, normalize=True)
    #             test_dataset = NormalizedDataset(X_test, y_test, normalize=True)

    #             clean_train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    #             clean_test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    #     else:
    #         logger.warning(f"Unsupported dataset format: {dataset_path}")

    #     return clean_train_loader, clean_test_loader


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
                clean_train_loader = DataLoader(clean_train_dataset, batch_size=512, shuffle=False)
                clean_test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long())
                clean_test_loader = DataLoader(clean_test_dataset, batch_size=512 , shuffle=False)
        else:
            logger.warning(f"{self.combination_id}: Unsupported dataset format: {dataset_path}")
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

    # def evaluate_pgd(self, model, loader):
    #     model.eval()
    #     atk = PGD(model, eps=8/255, alpha=2/255, steps=10)
    #     correct, total = 0, 0
    #     for X, y in loader:
    #         X, y = X.to(self.device), y.to(self.device)
    #         adv_X = atk(X, y)
    #         with torch.no_grad():
    #             pred = model(adv_X).argmax(1)
    #             correct += (pred == y).sum().item()
    #             total += y.size(0)
    #     return correct / total if total > 0 else 0.0

    #Replace pgd
    def evaluate_pgd(self, model, loader):
        import torch
        import torch.nn as nn
        from torchattacks import PGD
        from contextlib import nullcontext

        model.eval()

        # --- helpers kept local to avoid changing other files ---
        def _to_pixel_nchw(X: torch.Tensor) -> torch.Tensor:
            X = X.float()
            if X.max() > 1.5:  # handle uint8-like arrays
                X = X / 255.0
            if X.ndim == 4 and X.shape[1] != 3 and X.shape[-1] == 3:  # NHWC -> NCHW
                X = X.permute(0, 3, 1, 2).contiguous()
            return X

        class NormalizeWrapper(nn.Module):
            def __init__(self, base, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
                super().__init__()
                self.base = base
                m = torch.tensor(mean).view(1, 3, 1, 1)
                s = torch.tensor(std).view(1, 3, 1, 1)
                self.register_buffer("mean", m)
                self.register_buffer("std", s)
            def forward(self, x_pix):
                return self.base((x_pix - self.mean) / self.std)

        @torch.no_grad()
        def _probe_clean_acc(m_like: nn.Module, max_batches=3):
            m_like.eval()
            seen = correct = 0
            for bi, (Xb, yb) in enumerate(loader):
                if bi >= max_batches: break
                Xb = _to_pixel_nchw(Xb).to(self.device)
                yb = yb.to(self.device)
                pred = m_like(Xb).argmax(1)
                correct += (pred == yb).sum().item()
                seen += yb.size(0)
            return correct / max(seen, 1)

        # --- pick preprocessing once (no norm / 0.5 norm / CIFAR-10 stats) ---
        candidates = [
            (lambda m: m),  # pixel [0,1]
            (lambda m: NormalizeWrapper(m, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))),
            (lambda m: NormalizeWrapper(m, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))),
        ]
        best_model, best_acc = None, -1.0
        for wrap in candidates:
            wrapped = wrap(model).to(self.device)
            acc = _probe_clean_acc(wrapped, max_batches=3)
            if acc > best_acc:
                best_model, best_acc = wrapped, acc
        atk_model = best_model  # this model always expects pixel [0,1] inputs

        # --- your PGD-old settings ---
        atk = PGD(atk_model, eps=8/255, alpha=2/255, steps=10, random_start=True)

        correct, total = 0, 0
        # disable AMP during adversarial generation
        use_cuda = isinstance(self.device, torch.device) and self.device.type == "cuda"
        ctx = torch.amp.autocast("cuda", enabled=False) if use_cuda else nullcontext()

        for X, y in loader:
            X = _to_pixel_nchw(X).to(self.device)  # ensure pixel [0,1], NCHW
            y = y.to(self.device)
            with ctx:
                adv_X = atk(X, y)
            with torch.no_grad():
                pred = atk_model(adv_X).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        return correct / total if total > 0 else 0.0


    def evaluate_carlini_l2(self, model, loader, confidence=0):
        """
        Evaluate model against Carlini L2 attack (primary attack for MagNet)
        """
        try:
            from landseer_pipeline.evaluator.carlini_attack import evaluate_carlini_l2
            return evaluate_carlini_l2(model, loader, self.device, confidence)
        except ImportError:
            logger.warning("Carlini L2 attack not available")
            return -1.0
        
    @torch.no_grad()
    def margin_auc_from_loaders(
        self,
        model: torch.nn.Module,
        member_loader: torch.utils.data.DataLoader,
        nonmember_loader: torch.utils.data.DataLoader,
        device: Optional[torch.device] = None,
        # eps_dp: Optional[float] = None,   # accepted for compatibility, unused
        delta_dp: Optional[float] = 1e-5,  # choose δ to match your training claim
    ) -> Tuple[float, float]:
        """
        Returns:
            (auc_margin, eps_hat)
          - auc_margin: AUROC of margin (p_top1 - p_top2)
          - eps_hat: empirical lower bound on ε at δ=delta_dp (>= 0)
        """

        was_training = model.training
        model.eval()

        def collect_margins(loader):
            vals = []
            for x, _ in loader:
                x = x.to(self.device, non_blocking=True)
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                top2 = probs.topk(2, dim=1).values
                margin = (top2[:, 0] - top2[:, 1]).detach().cpu().numpy()
                vals.append(margin)
            return np.concatenate(vals, axis=0) if vals else np.array([])

        m_scores = collect_margins(member_loader)
        nm_scores = collect_margins(nonmember_loader)

        if was_training:
            model.train()

        if m_scores.size == 0 or nm_scores.size == 0:
            return float("nan"), 0.0

        y_true = np.concatenate([np.ones_like(m_scores, dtype=int),
                                 np.zeros_like(nm_scores, dtype=int)])
        y_scores = np.concatenate([m_scores, nm_scores])

        # AUROC
        auc_margin = float(roc_auc_score(y_true, y_scores))

        # ε lower bound via DP hypothesis testing inequalities
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        delta = 1e-5 if (delta_dp is None) else float(delta_dp)
        eps_vals = []
        tiny = 1e-12
        for a, b in zip(fpr, tpr):
            # β ≤ e^ε α + δ  -> ε ≥ ln((β-δ)/α)
            if a > tiny and (b - delta) > tiny:
                eps_vals.append(np.log((b - delta) / a))
            # (1-β) ≤ e^ε (1-α) + δ -> ε ≥ ln(((1-β)-δ)/(1-α))
            if (1 - a) > tiny and ((1 - b) - delta) > tiny:
                eps_vals.append(np.log(((1 - b) - delta) / (1 - a)))

        eps_hat = max(eps_vals) if eps_vals else float("-inf")
        if not np.isfinite(eps_hat) or eps_hat < 0.0:
            eps_hat = 0.0  # ε is nonnegative by definition

        return auc_margin, float(eps_hat)

    def evaluate_outlier(self, model, clean_loader, ood_loader):
        model.eval()
        scores, labels = [], []
        
        # Use mixed precision and non-blocking transfers
        device_obj = torch.device(self.device) if isinstance(self.device, str) else self.device
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device_obj.type == "cuda"):
            # Process clean samples
            for X, _ in clean_loader:
                X = X.to(self.device, non_blocking=True)
                out = torch.softmax(model(X), dim=1)
                max_conf = out.max(1)[0]
                scores.extend((1 - max_conf).cpu().numpy())
                labels.extend([0] * X.size(0))
            
            # Process OOD samples
            for X, _ in ood_loader:
                X = X.to(self.device, non_blocking=True)
                out = torch.softmax(model(X), dim=1)
                max_conf = out.max(1)[0]
                scores.extend((1 - max_conf).cpu().numpy())
                labels.extend([1] * X.size(0))
        
        return roc_auc_score(labels, scores)

    def evaluate_backdoor(self, model, poisoned_loader, target_class):
        model.eval()
        total, target_hits = 0, 0
        
        # Use mixed precision and non-blocking transfers
        device_obj = torch.device(self.device) if isinstance(self.device, str) else self.device
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device_obj.type == "cuda"):
            for X, _ in poisoned_loader:
                X = X.to(self.device)
                pred = model(X).argmax(1)
                target_hits += (pred == target_class).sum().item()
                total += X.size(0)
        return target_hits / total

    def find_uchida_style_conv2d(self, model):
        """Find appropriate Conv2D layer for watermark embedding (adapted from watermarking.py)"""
        conv_layers = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_layers.append((name, module))
        
        if not conv_layers:
            return None

        # Prefer second-last conv if exists, else fallback to last
        if len(conv_layers) >= 2:
            selected_idx = -2
        else:
            selected_idx = -1

        selected_name, selected_layer = conv_layers[selected_idx]
        logger.info(f"{self.combination_id}: Selected Conv2D layer for watermark evaluation: {selected_name}")
        return selected_layer

    def decode_watermark(self, conv_layer, wm_matrix, wm_bits):
        """Decode watermark from Conv2D layer (adapted from watermarking.py)"""
        #with torch.no_grad():
            #weight = conv_layer.weight
            #mean_weight = weight.mean(dim=[2, 3])
            #flat = mean_weight.view(1, -1)
            #projection = torch.sigmoid(flat @ wm_matrix)
            #decoded_bits = (projection > 0.5).float()
            #accuracy = (decoded_bits == wm_bits).float().mean().item()
            #return accuracy, decoded_bits.cpu().numpy()
        try:
            with torch.no_grad():
                weight = conv_layer.weight
                mean_weight = weight.mean(dim=[2, 3])
                flat = mean_weight.view(1, -1)
                # Check dimension compatibility
                if flat.shape[1] != wm_matrix.shape[0]:
                    logger.warning(f"{self.combination_id}: Dimension mismatch: flat shape {flat.shape} vs wm_matrix shape {wm_matrix.shape}")
                    # Try to reshape or truncate to match
                    expected_dim = wm_matrix.shape[0]
                    if flat.shape[1] > expected_dim:
                        flat = flat[:, :expected_dim]
                    else:
                        # Pad with zeros if too small
                        padding_size = expected_dim - flat.shape[1]
                        flat = torch.cat([flat, torch.zeros(1, padding_size, device=flat.device)], dim=1)
                projection = torch.sigmoid(flat @ wm_matrix)
                decoded_bits = (projection > 0.5).float
                accuracy = (decoded_bits == wm_bits).float().mean().item()
                return accuracy, decoded_bits.cpu().numpy()
        except Exception as e:
            logger.error(f"{self.combination_id}: Error in watermark decoding: {e}")
        return 0.0, None
        return 0.0, None

    def evaluate_watermark(self, model) -> float:
        """Evaluate watermark detection accuracy - supports both weight-based and trigger-based watermarking"""
        # Check for weight-based watermarking (existing tool)
        wm_matrix_path = self.input_wm_matrix
        wm_bits_path = self.input_wm_bits
        
        if wm_matrix_path and wm_bits_path and os.path.exists(wm_matrix_path) and os.path.exists(wm_bits_path):
            return self._evaluate_weight_based_watermark(model, wm_matrix_path, wm_bits_path)
        
        # Check for trigger-based watermarking (WatermarkNN)
        watermark_trigger_accuracy = self._evaluate_trigger_based_watermark(model)
        if watermark_trigger_accuracy is not None:
            return watermark_trigger_accuracy

        logger.warning(f"{self.combination_id}: No watermarking evaluation method found - neither weight-based nor trigger-based")
        return 0.0
    
    def _evaluate_weight_based_watermark(self, model, wm_matrix_path: str, wm_bits_path: str) -> float:
        """Evaluate weight-based watermarking (existing watermark tool)"""
        try:
            logger.info(f"{self.combination_id}: Evaluating weight-based watermarking...")
            # Load watermark matrix and bits
            wm_matrix = torch.tensor(np.load(wm_matrix_path), dtype=torch.float32).to(self.device)
            wm_bits = torch.tensor(np.load(wm_bits_path), dtype=torch.float32).to(self.device)
            
            # Find appropriate Conv2D layer
            conv_layer = self.find_uchida_style_conv2d(model)
            if conv_layer is None:
                logger.warning(f"{self.combination_id}: No Conv2D layer found for watermark evaluation")
                return 0.0
            
            # Decode watermark
            accuracy, _ = self.decode_watermark(conv_layer, wm_matrix, wm_bits)
            logger.info(f"{self.combination_id}: Weight-based watermark decoding accuracy: {accuracy * 100:.2f}%")
            return accuracy
            
        except Exception as e:
            logger.error(f"{self.combination_id}: Error during weight-based watermark evaluation: {e}")
            return 0.0
    
    def _evaluate_trigger_based_watermark(self, model) -> float:
        """Evaluate trigger-based watermarking (WatermarkNN style)"""
        try:
            # Look for watermark triggers directory and training summary
            watermark_triggers_path = self._find_watermark_triggers_path()
            training_summary_path = self._find_training_summary_path()
            
            if not watermark_triggers_path:
                logger.debug(f"{self.combination_id}: No watermark triggers found - not trigger-based watermarking")
                return None

            logger.info(f"{self.combination_id}: Evaluating trigger-based watermarking using triggers from: {watermark_triggers_path}")

            # Load watermark trigger data
            watermark_loader = self._load_watermark_triggers(watermark_triggers_path)
            if watermark_loader is None:
                logger.warning(f"{self.combination_id}: Failed to load watermark triggers")
                return 0.0
            
            # Evaluate watermark trigger accuracy
            model.eval()
            correct, total = 0, 0
            
            with torch.no_grad():
                for data, targets in watermark_loader:
                    data, targets = data.to(self.device), targets.to(self.device)
                    outputs = model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            
            watermark_accuracy = correct / total if total > 0 else 0.0
            logger.info(f"{self.combination_id}: Trigger-based watermark accuracy: {watermark_accuracy * 100:.2f}%")
            return watermark_accuracy
            
        except Exception as e:
            logger.error(f"{self.combination_id}: Error during trigger-based watermark evaluation: {e}")
            return None
    
    def _find_watermark_triggers_path(self) -> str:
        """Find watermark triggers directory in combination output"""
        # Check combination output directory for watermark_triggers folder
        combination_dir = Path(self.combination_output)
        
        # Look for watermark_triggers in the final tool output
        watermark_triggers_dir = combination_dir / "watermark_triggers"
        if watermark_triggers_dir.exists() and watermark_triggers_dir.is_dir():
            return str(watermark_triggers_dir)
        
        return None
    
    def _find_training_summary_path(self) -> str:
        """Find training summary JSON file"""
        combination_dir = Path(self.combination_output)
        summary_path = combination_dir / "training_summary.json"
        return str(summary_path) if summary_path.exists() else None
    
    def _load_watermark_triggers(self, watermark_triggers_path: str):
        """Load watermark trigger images and labels for evaluation"""
        try:
            # Use the same transforms as WatermarkNN
            transform_wm = transforms.Compose([
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            
            # Check for labels file (support multiple naming conventions)
            labels_file = None
            for labels_name in ["labels.txt", "labels-cifar.txt"]:
                potential_path = os.path.join(watermark_triggers_path, labels_name)
                if os.path.exists(potential_path):
                    labels_file = potential_path
                    break
            
            if not labels_file:
                logger.warning(f"{self.combination_id}: Labels file not found in: {watermark_triggers_path}")
                return None
            
            # Load labels
            wm_targets = np.loadtxt(labels_file)
            logger.info(f"Loaded {len(wm_targets)} watermark trigger labels from {labels_file}")
            
            # Create custom dataset similar to WatermarkNN's approach
            class WatermarkTriggerDataset(torch.utils.data.Dataset):
                def __init__(self, root_dir, targets, transform=None):
                    self.root_dir = root_dir
                    self.targets = targets
                    self.transform = transform
                    
                    # Get all image files (check both root and pics subdirectory)
                    import glob
                    self.image_paths = []
                    
                    # First check pics subdirectory (WatermarkNN format)
                    pics_dir = os.path.join(root_dir, "pics")
                    if os.path.exists(pics_dir):
                        search_dir = pics_dir
                    else:
                        search_dir = root_dir
                    
                    for ext in ['*.png', '*.jpg', '*.jpeg']:
                        self.image_paths.extend(glob.glob(os.path.join(search_dir, ext)))
                    self.image_paths.sort()  # Ensure consistent ordering
                
                def __len__(self):
                    return min(len(self.image_paths), len(self.targets))
                
                def __getitem__(self, idx):
                    from PIL import Image
                    img_path = self.image_paths[idx]
                    image = Image.open(img_path).convert('RGB')
                    
                    if self.transform:
                        image = self.transform(image)
                    
                    target = int(self.targets[idx])
                    return image, target
            
            dataset = WatermarkTriggerDataset(watermark_triggers_path, wm_targets, transform_wm)
            
            if len(dataset) == 0:
                logger.warning(f"{self.combination_id}: No watermark trigger images found")
                return None
            
            watermark_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
            logger.info(f"{self.combination_id}: Loaded {len(dataset)} watermark trigger samples")
            return watermark_loader
            
        except Exception as e:
            logger.error(f"Error loading watermark triggers: {e}")
            return None
    
    # def generate_ood_samples(self, dataset_path: str):
    #     ood_loader = None
    #     if dataset_path:
    #         files = os.listdir(dataset_path)
    #         if 'test_data.npy' in files and 'test_labels.npy' in files:
    #             X_train = np.load(os.path.join(dataset_path, 'data.npy'))
    #             y_train = np.load(os.path.join(dataset_path, 'labels.npy'))
    #             X_test = np.load(os.path.join(dataset_path, 'test_data.npy'))
    #             y_test = np.load(os.path.join(dataset_path, 'test_labels.npy'))
    #             X_train = torch.tensor(X_train).float()
    #             y_train = torch.tensor(y_train).long()
    #             X_test = torch.tensor(X_test).float()
    #             y_test = torch.tensor(y_test).long()
    #             ood_loader = DataLoader(TensorDataset(torch.rand_like(X_test), y_test), batch_size=64, shuffle=False)
    #     return ood_loader

    def generate_ood_samples(self, dataset_path: str):
        """
        Load a selected OOD dataset for CIFAR-10 evaluation.
        
        dataset_path: str — one of ['cifar100', 'svhn', 'tinyimagenet']
        """
        ood_loader = None

        # CIFAR-10 normalization
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                std=(0.2023, 0.1994, 0.2010)),
        ])

        #if dataset_path.lower() == 'cifar10':
        ood_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        #elif dataset_path.lower() == 'svhn':
        #    ood_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=transform)

        #else:
        #    raise ValueError(f"Unsupported OOD dataset: {dataset_path}. Choose from ['cifar100', 'svhn']")

        ood_loader = DataLoader(ood_dataset, batch_size=64, shuffle=False)
        return ood_loader
    
    def extract_privacy_epsilon(self) -> float:
        """Extract privacy epsilon from privacy_metrics.txt file specified in fin_output_paths.json"""
        json_path = Path(self.combination_output) / "fin_output_paths.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                paths = json.load(f)
            
            # Look for privacy_metrics.txt in the paths
            privacy_entry = paths.get("privacy_metrics.txt", {})
            if isinstance(privacy_entry, dict):
                privacy_file_path = privacy_entry.get("source_path", "")
            else:
                privacy_file_path = ""
            if privacy_file_path and os.path.exists(privacy_file_path):
                epsilon = self._parse_privacy_params(Path(privacy_file_path))
                if epsilon >= 0:
                    logger.info(f"{self.combination_id}: Found privacy epsilon: {epsilon} in {privacy_file_path}")
                    return epsilon

        logger.warning(f"{self.combination_id}: No privacy metrics file found in fin_output_paths.json or file does not exist.")
        return -1.0
        
    def _parse_privacy_params(self, params_file: Path) -> float:
        # epsilon=3.0
        # delta=1e-05
        # dp_accuracy=0.1017
        try:
            with open(params_file, 'r') as f:
                for line in f:
                    if line.startswith("epsilon="):
                        return float(line.split('=')[1].strip())
        except Exception as e:
            logger.warning(f"Could not parse privacy parameters from {params_file}: {e}")
        return -1.0
    
    def extract_dp_accuracy(self) -> float:
        """Extract DP accuracy from privacy_metrics.txt file specified in fin_output_paths.json"""
        json_path = Path(self.combination_output) / "fin_output_paths.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                paths = json.load(f)
            
            # Look for privacy_metrics.txt in the paths
            privacy_file_path = paths.get("privacy_metrics.txt", "").get("source_path", "")
            if privacy_file_path and os.path.exists(privacy_file_path):
                acc = self._get_dp_accuracy(Path(privacy_file_path))
                if acc >= 0:
                    logger.info(f"Found DP accuracy: {acc} in {privacy_file_path}")
                    return acc
        
        logger.warning("No privacy metrics file found in fin_output_paths.json or file does not exist.")
        return -1.0
        
    def _get_dp_accuracy(self, params_file: Path) -> float:
        # epsilon=3.0
        # delta=1e-05
        # dp_accuracy=0.1017
        try:
            with open(params_file, 'r') as f:
                for line in f:
                    if line.startswith("dp_accuracy="):
                        return float(line.split('=')[1].strip())
        except Exception as e:
            logger.warning(f"Could not parse privacy parameters from {params_file}: {e}")
        return -1.0