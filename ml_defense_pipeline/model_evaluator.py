"""
Model evaluation for ML Defense Pipeline
"""
import logging
import os
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
    
    def train_baseline_model(self, dataset_path: str, output_path: str) -> float:
        """
        Train a baseline model for comparison
        
        Args:
            dataset_path: Path to dataset in .h5 format
            output_path: Path to save the baseline model
            
        Returns:
            Baseline accuracy as a float
        """
        logger.info("Training baseline model for comparison...")
        
        try:
            import h5py
            import tensorflow as tf
            import numpy as np
            
            # Load data
            with h5py.File(dataset_path, 'r') as ds:
                X_train = ds["X_train"][:]
                y_train = ds["y_train"][:]
                X_test = ds["X_test"][:]
                y_test = ds["y_test"][:]
            
            # Create a simple CNN model
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
            
            # Compile model
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train model
            model.fit(
                X_train, y_train,
                epochs=5,
                batch_size=64,
                validation_data=(X_test, y_test),
                verbose=1
            )
            
            # Evaluate model
            _, acc = model.evaluate(X_test, y_test, verbose=0)
            
            # Save model
            model.save(output_path)
            
            logger.info(f"Baseline model trained with accuracy: {acc:.4f}")
            return acc
            
        except ImportError as e:
            logger.warning(f"Could not train baseline model: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Error training baseline model: {e}")
            return 0.0