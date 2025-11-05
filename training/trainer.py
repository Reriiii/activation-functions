"""
Training module for running experiments
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import os
from pathlib import Path


class Trainer:
    """
    Trainer class for managing model training
    """
    
    def __init__(self, model, train_dataset, val_dataset, test_dataset, 
                 experiment_name, save_dir='results'):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.experiment_name = experiment_name
        self.save_dir = Path(save_dir)
        
        # Create directories
        self.model_dir = self.save_dir / 'models' / experiment_name
        self.log_dir = self.save_dir / 'logs' / experiment_name
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = None
        self.training_time = 0
        
    def train(self, epochs=10, callbacks=None):
        """
        Train the model
        
        Args:
            epochs: Number of epochs to train
            callbacks: List of callbacks
            
        Returns:
            Training history
        """
        # Default callbacks
        default_callbacks = [
            keras.callbacks.ModelCheckpoint(
                str(self.model_dir / 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.TensorBoard(
                log_dir=str(self.log_dir),
                histogram_freq=1,
                write_graph=True
            ),
            keras.callbacks.CSVLogger(
                str(self.log_dir / 'training.log')
            )
        ]
        
        if callbacks:
            default_callbacks.extend(callbacks)
        
        # Train model
        print(f"\n{'='*60}")
        print(f"Training {self.experiment_name}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        self.history = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=epochs,
            callbacks=default_callbacks,
            verbose=1
        )
        
        self.training_time = time.time() - start_time
        
        print(f"\nTraining completed in {self.training_time:.2f} seconds")
        
        return self.history
    
    def evaluate(self):
        """
        Evaluate model on test set
        
        Returns:
            Dictionary with test metrics
        """
        print(f"\n{'='*60}")
        print(f"Evaluating {self.experiment_name}")
        print(f"{'='*60}\n")
        
        test_results = self.model.evaluate(self.test_dataset, verbose=1)
        
        # Create results dictionary
        results = {}
        for metric_name, metric_value in zip(self.model.metrics_names, test_results):
            results[f'test_{metric_name}'] = metric_value
            
        results['training_time'] = self.training_time
        
        return results
    
    def predict(self, dataset):
        """
        Make predictions on a dataset
        
        Args:
            dataset: tf.data.Dataset
            
        Returns:
            predictions: numpy array of predictions
        """
        predictions = self.model.predict(dataset, verbose=1)
        return predictions
    
    def save_model(self, filename='final_model.h5'):
        """Save the trained model"""
        save_path = self.model_dir / filename
        self.model.save(str(save_path))
        print(f"Model saved to {save_path}")
        
    def load_model(self, filename='best_model.h5'):
        """Load a saved model"""
        load_path = self.model_dir / filename
        self.model = keras.models.load_model(str(load_path))
        print(f"Model loaded from {load_path}")
        return self.model
    
    def get_training_history(self):
        """Return training history as dictionary"""
        if self.history is None:
            return None
            
        return {
            'loss': self.history.history['loss'],
            'accuracy': self.history.history['accuracy'],
            'val_loss': self.history.history['val_loss'],
            'val_accuracy': self.history.history['val_accuracy'],
            'epochs': len(self.history.history['loss'])
        }


class ExperimentTracker:
    """
    Track multiple experiments and compare results
    """
    
    def __init__(self, save_dir='results'):
        self.save_dir = Path(save_dir)
        self.experiments = {}
        
    def add_experiment(self, name, results):
        """Add experiment results"""
        self.experiments[name] = results
        
    def save_results(self, filename='experiment_results.csv'):
        """Save all experiment results to CSV"""
        import pandas as pd
        
        df = pd.DataFrame(self.experiments).T
        save_path = self.save_dir / filename
        df.to_csv(save_path)
        print(f"Results saved to {save_path}")
        
        return df
    
    def get_best_experiment(self, metric='test_accuracy'):
        """Get the best performing experiment"""
        if not self.experiments:
            return None
            
        best_name = max(self.experiments.keys(), 
                       key=lambda k: self.experiments[k].get(metric, 0))
        
        return best_name, self.experiments[best_name]
    
    def compare_experiments(self):
        """Print comparison of all experiments"""
        if not self.experiments:
            print("No experiments to compare")
            return
            
        print("\n" + "="*80)
        print("EXPERIMENT COMPARISON")
        print("="*80)
        
        for name, results in self.experiments.items():
            print(f"\n{name}:")
            for metric, value in results.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")