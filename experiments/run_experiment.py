"""
Main experiment runner
"""
import sys
import os
import yaml
import argparse
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.data_loader import DataLoader
from models.alexnet import AlexNet, AlexNetSmall
from models.vgg16 import VGG16, VGG16Small
from models.resnet import ResNet18, ResNet34
from models.efficientnet import EfficientNetSmall
from training.trainer import Trainer, ExperimentTracker
from evaluation.visualization import Visualizer


class ExperimentRunner:
    """
    Run experiments with different configurations
    """
    
    def __init__(self, config_file='experiment_config.yaml'):
        self.config_file = config_file
        self.config = self._load_config()
        self.tracker = ExperimentTracker(save_dir=self.config['save_dir'])
        self.visualizer = Visualizer(save_dir=f"{self.config['save_dir']}/plots")
        
    def _load_config(self):
        """Load configuration from YAML file"""
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _get_model(self, model_name, input_shape, num_classes, activation):
        """Get model instance by name"""
        models = {
            'alexnet': AlexNet,
            'alexnet_small': AlexNetSmall,
            'vgg16': VGG16,
            'vgg16_small': VGG16Small,
            'resnet18': ResNet18,
            'resnet34': ResNet34,
            'efficientnet_small': EfficientNetSmall,
        }
        
        model_class = models.get(model_name.lower())
        if model_class is None:
            raise ValueError(f"Model {model_name} not supported")
        
        return model_class(input_shape, num_classes, activation)
    
    def run_single_experiment(self, dataset_name, model_name, activation_name):
        """
        Run a single experiment
        
        Args:
            dataset_name: Name of dataset
            model_name: Name of model
            activation_name: Name of activation function
            
        Returns:
            Dictionary with experiment results
        """
        experiment_name = f"{dataset_name}_{model_name}_{activation_name}"
        
        print(f"\n{'='*80}")
        print(f"Running Experiment: {experiment_name}")
        print(f"{'='*80}\n")
        
        # Load data
        data_loader = DataLoader(
            dataset_name=dataset_name,
            batch_size=self.config['batch_size'],
            validation_split=self.config['validation_split']
        )
        train_ds, val_ds, test_ds = data_loader.load_data()
        data_info = data_loader.get_data_info()
        
        # Create model
        model_builder = self._get_model(
            model_name,
            data_info['input_shape'],
            data_info['num_classes'],
            activation_name
        )
        model_builder.build_model()
        model_builder.compile_model(
            optimizer=self.config['optimizer'],
            learning_rate=self.config['learning_rate']
        )
        
        # Print model summary
        print("\nModel Summary:")
        model_builder.summary()
        params = model_builder.count_parameters()
        print(f"\nTotal Parameters: {params['total']:,}")
        print(f"Trainable Parameters: {params['trainable']:,}")
        
        # Train model
        trainer = Trainer(
            model=model_builder.get_model(),
            train_dataset=train_ds,
            val_dataset=val_ds,
            test_dataset=test_ds,
            experiment_name=experiment_name,
            save_dir=self.config['save_dir']
        )
        
        history = trainer.train(epochs=self.config['epochs'])
        
        # Evaluate model
        test_results = trainer.evaluate()
        
        # Visualize training history
        self.visualizer.plot_training_history(
            history,
            title=f'{experiment_name} - Training History',
            filename=f'{experiment_name}_history.png'
        )
        
        # Compile results
        results = {
            'experiment_name': experiment_name,
            'dataset': dataset_name,
            'model': model_name,
            'activation': activation_name,
            **test_results,
            'total_params': params['total'],
            'trainable_params': params['trainable']
        }
        
        # Add to tracker
        self.tracker.add_experiment(experiment_name, results)
        
        return results
    
    def _scan_completed_experiments_strict(self):
            """
            Kiểm tra nghiêm ngặt 3 điều kiện.
            Chỉ return thí nghiệm nào có đủ: Model (.h5) + Log (.log) + Plot (.png)
            """
            completed = set()
            
            base_dir = self.config['save_dir']
            models_dir = os.path.join(base_dir, 'models')
            logs_dir = os.path.join(base_dir, 'logs')
            plots_dir = os.path.join(base_dir, 'plots')

            print(f"[INFO] Strict Scanning (Model + Log + Plot) in {base_dir}...")

            if not os.path.exists(models_dir):
                return completed

            # Duyệt qua danh sách thư mục trong models
            for exp_name in os.listdir(models_dir):
                
                # 1. Kiểm tra MODEL: results/models/<exp_name>/best_model.h5
                model_path = os.path.join(models_dir, exp_name, 'best_model.h5')
                if not os.path.isfile(model_path):
                    continue # Thiếu model -> Bỏ qua

                # 2. Kiểm tra LOG: results/logs/<exp_name>/training.log
                log_path = os.path.join(logs_dir, exp_name, 'training.log')
                if not os.path.isfile(log_path):
                    continue # Thiếu log -> Bỏ qua

                # 3. Kiểm tra PLOT: results/plots/<exp_name>_history.png
                # Lưu ý: Dựa vào file tree bạn gửi, plots nằm thẳng trong folder plots, ko có sub-folder
                plot_path = os.path.join(plots_dir, f"{exp_name}_history.png")
                if not os.path.isfile(plot_path):
                    continue # Thiếu plot -> Bỏ qua

                # Nếu đủ cả 3 -> OK
                completed.add(exp_name)
            
            return completed
        
    def run_all_experiments(self):
        """Run all experiments defined in config"""
        
        all_results = []
        
        completed_experiments = self._scan_completed_experiments_strict()
        
        print(f"\n[INFO] RECOVERY MODE: Found {len(completed_experiments)} completed experiments on disk.")
        
        datasets = self.config['datasets']
        models = self.config['models']
        activations = self.config['activations']
        
        total_experiments = len(datasets) * len(models) * len(activations)
        current = 0
        
        print(f"\n{'='*80}")
        print(f"Starting {total_experiments} experiments")
        print(f"{'='*80}\n")
        
        for dataset in datasets:
            for model in models:
                for activation in activations:
                    current += 1
                    
                    experiment_name = f"{dataset}_{model}_{activation}"
                    
                    if experiment_name in completed_experiments:
                        print(f"Progress: {current}/{total_experiments} - [SKIP] {experiment_name}")
                        continue
                    
                    print(f"\nProgress: {current}/{total_experiments}")
                    

                    try:
                        results = self.run_single_experiment(dataset, model, activation)
                        all_results.append(results)
                    except Exception as e:
                        print(f"Error in experiment {dataset}_{model}_{activation}: {str(e)}")
                        continue
        
        # Save all results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(f"{self.config['save_dir']}/all_results.csv", index=False)
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        self._generate_visualizations(results_df)
        
        # Print summary
        self._print_summary(results_df)
        
        return results_df
    
    def _generate_visualizations(self, results_df):
        """Generate all visualization plots"""
        
        # Overall comparison
        results_dict = results_df.set_index('experiment_name')['test_accuracy'].to_dict()
        self.visualizer.plot_comparison(
            {k: {'test_accuracy': v} for k, v in results_dict.items()},
            metric='test_accuracy',
            title='Test Accuracy Comparison - All Experiments',
            filename='all_experiments_comparison.png'
        )
        
        # Activation comparison for each model/dataset combination
        for dataset in results_df['dataset'].unique():
            for model in results_df['model'].unique():
                try:
                    self.visualizer.plot_activation_comparison(
                        results_df,
                        dataset_name=dataset,
                        model_name=model,
                        filename=f'{dataset}_{model}_activation_comparison.png'
                    )
                except:
                    continue
        
        # Heatmap
        self.visualizer.plot_heatmap_comparison(
            results_df,
            metric='test_accuracy',
            filename='accuracy_heatmap.png'
        )
        
        # Comprehensive report
        self.visualizer.create_report(
            results_df,
            output_file='comprehensive_report.png'
        )
    
    def _print_summary(self, results_df):
        """Print experiment summary"""
        
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        
        print("\n1. Best Overall Performance:")
        best_idx = results_df['test_accuracy'].idxmax()
        best = results_df.loc[best_idx]
        print(f"   Experiment: {best['experiment_name']}")
        print(f"   Test Accuracy: {best['test_accuracy']:.4f}")
        print(f"   Model: {best['model']}")
        print(f"   Activation: {best['activation']}")
        print(f"   Dataset: {best['dataset']}")
        
        print("\n2. Best Activation Function (Average):")
        act_avg = results_df.groupby('activation')['test_accuracy'].mean().sort_values(ascending=False)
        for act, acc in act_avg.head(3).items():
            print(f"   {act}: {acc:.4f}")
        
        print("\n3. Best Model Architecture (Average):")
        model_avg = results_df.groupby('model')['test_accuracy'].mean().sort_values(ascending=False)
        for model, acc in model_avg.head(3).items():
            print(f"   {model}: {acc:.4f}")
        
        print("\n4. Training Time Analysis:")
        time_avg = results_df.groupby('model')['training_time'].mean().sort_values()
        print(f"   Fastest: {time_avg.index[0]} ({time_avg.iloc[0]:.2f}s)")
        print(f"   Slowest: {time_avg.index[-1]} ({time_avg.iloc[-1]:.2f}s)")
        
        print("\n" + "="*80 + "\n")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run activation function experiments')
    parser.add_argument('--config', type=str, default='experiment_config.yaml',
                       help='Path to config file')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Run experiment for specific dataset')
    parser.add_argument('--model', type=str, default=None,
                       help='Run experiment for specific model')
    parser.add_argument('--activation', type=str, default=None,
                       help='Run experiment for specific activation')
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(config_file=args.config)
    
    # Run single experiment if specific parameters provided
    if args.dataset and args.model and args.activation:
        runner.run_single_experiment(args.dataset, args.model, args.activation)
    else:
        # Run all experiments
        runner.run_all_experiments()


if __name__ == '__main__':
    main()