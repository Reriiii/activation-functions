"""
Visualization utilities for experiments
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path


class Visualizer:
    """
    Visualization class for experiment results
    """
    
    def __init__(self, save_dir='results/plots'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)
        
    def plot_training_history(self, history, title='Training History', 
                              filename='training_history.png'):
        """
        Plot training and validation metrics
        
        Args:
            history: Training history dictionary or History object
            title: Plot title
            filename: Filename to save plot
        """
        if hasattr(history, 'history'):
            history = history.history
            
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        axes[0].plot(history['accuracy'], label='Training')
        axes[0].plot(history['val_accuracy'], label='Validation')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot loss
        axes[1].plot(history['loss'], label='Training')
        axes[1].plot(history['val_loss'], label='Validation')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        save_path = self.save_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.close()
        
    def plot_comparison(self, results_dict, metric='test_accuracy', 
                       title='Model Comparison', filename='comparison.png'):
        """
        Plot comparison of multiple experiments
        
        Args:
            results_dict: Dictionary of {experiment_name: results}
            metric: Metric to compare
            title: Plot title
            filename: Filename to save plot
        """
        names = list(results_dict.keys())
        values = [results_dict[name][metric] for name in names]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.bar(range(len(names)), values, color=sns.color_palette('husl', len(names)))
        
        ax.set_xlabel('Experiment')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(title)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}',
                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        save_path = self.save_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.close()
        
    def plot_activation_comparison(self, results_df, dataset_name='MNIST', 
                                   model_name='AlexNet', 
                                   filename='activation_comparison.png'):
        """
        Plot comparison of different activation functions
        
        Args:
            results_df: DataFrame with experiment results
            dataset_name: Name of dataset
            model_name: Name of model
            filename: Filename to save plot
        """
        # Filter results for specific model and dataset
        mask = (results_df['model'] == model_name) & (results_df['dataset'] == dataset_name)
        filtered_df = results_df[mask]
        
        if filtered_df.empty:
            print(f"No results found for {model_name} on {dataset_name}")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = ['test_accuracy', 'test_loss', 'training_time', 'test_top5_accuracy']
        titles = ['Test Accuracy', 'Test Loss', 'Training Time (s)', 'Top-5 Accuracy']
        
        for ax, metric, title in zip(axes.flat, metrics, titles):
            if metric in filtered_df.columns:
                data = filtered_df.sort_values(metric, ascending=(metric == 'test_loss'))
                
                bars = ax.barh(data['activation'], data[metric], 
                              color=sns.color_palette('viridis', len(data)))
                
                ax.set_xlabel(title)
                ax.set_ylabel('Activation Function')
                ax.set_title(f'{title} - {model_name} on {dataset_name}')
                
                # Add value labels
                for i, (bar, val) in enumerate(zip(bars, data[metric])):
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height()/2.,
                           f'{val:.4f}' if metric != 'training_time' else f'{val:.2f}s',
                           ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        
        save_path = self.save_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.close()
        
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None,
                             title='Confusion Matrix', filename='confusion_matrix.png'):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            title: Plot title
            filename: Filename to save plot
        """
        from sklearn.metrics import confusion_matrix
        
        # Convert one-hot to labels if needed
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        save_path = self.save_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.close()
        
    def plot_heatmap_comparison(self, results_df, metric='test_accuracy',
                               filename='heatmap_comparison.png'):
        """
        Plot heatmap comparing models and activations
        
        Args:
            results_df: DataFrame with experiment results
            metric: Metric to compare
            filename: Filename to save plot
        """
        # Pivot table for heatmap
        pivot_data = results_df.pivot_table(
            values=metric,
            index='model',
            columns='activation',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='YlGnBu',
                   cbar_kws={'label': metric.replace('_', ' ').title()})
        
        plt.title(f'{metric.replace("_", " ").title()} - Model vs Activation')
        plt.xlabel('Activation Function')
        plt.ylabel('Model Architecture')
        plt.tight_layout()
        
        save_path = self.save_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.close()
        
    def create_report(self, results_df, output_file='report.png'):
        """
        Create comprehensive visualization report
        
        Args:
            results_df: DataFrame with all experiment results
            output_file: Filename to save report
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Overall accuracy comparison
        ax1 = fig.add_subplot(gs[0, :])
        results_sorted = results_df.sort_values('test_accuracy', ascending=False).head(10)
        ax1.barh(results_sorted['experiment_name'], results_sorted['test_accuracy'])
        ax1.set_xlabel('Test Accuracy')
        ax1.set_title('Top 10 Experiments by Accuracy')
        
        # 2. Activation function performance
        ax2 = fig.add_subplot(gs[1, 0])
        act_perf = results_df.groupby('activation')['test_accuracy'].mean().sort_values()
        ax2.barh(act_perf.index, act_perf.values)
        ax2.set_xlabel('Average Test Accuracy')
        ax2.set_title('Activation Function Performance')
        
        # 3. Model performance
        ax3 = fig.add_subplot(gs[1, 1])
        model_perf = results_df.groupby('model')['test_accuracy'].mean().sort_values()
        ax3.barh(model_perf.index, model_perf.values)
        ax3.set_xlabel('Average Test Accuracy')
        ax3.set_title('Model Architecture Performance')
        
        # 4. Training time comparison
        ax4 = fig.add_subplot(gs[1, 2])
        time_comp = results_df.groupby('model')['training_time'].mean().sort_values()
        ax4.barh(time_comp.index, time_comp.values)
        ax4.set_xlabel('Training Time (s)')
        ax4.set_title('Average Training Time by Model')
        
        # 5. Heatmap
        ax5 = fig.add_subplot(gs[2, :])
        pivot_data = results_df.pivot_table(
            values='test_accuracy',
            index='model',
            columns='activation',
            aggfunc='mean'
        )
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax5)
        ax5.set_title('Test Accuracy Heatmap: Model vs Activation')
        
        plt.suptitle('Activation Function Research - Comprehensive Report', 
                    fontsize=16, y=0.995)
        
        save_path = self.save_dir / output_file
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Report saved to {save_path}")
        plt.close()