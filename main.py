import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

import tensorflow as tf
import argparse
from pathlib import Path

from experiments.run_experiment import ExperimentRunner


def setup_gpu():
    """Setup GPU configuration"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU found, using CPU")


def create_directories():
    """Create necessary directories"""
    dirs = [
        'data',
        'results/models',
        'results/logs',
        'results/plots',
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Activation Function Research Project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments
  python main.py
  
  # Run single experiment
  python main.py --dataset mnist --model alexnet_small --activation relu
  
  # Use custom config file
  python main.py --config my_config.yaml
  
  # Quick test (1 epoch)
  python main.py --quick-test
        """
    )
    
    parser.add_argument('--config', type=str, 
                       default='experiments/experiment_config.yaml',
                       help='Path to configuration file')
    
    parser.add_argument('--dataset', type=str, default=None,
                       choices=['mnist', 'fashion_mnist', 'cifar10', 'cifar100'],
                       help='Specific dataset to use')
    
    parser.add_argument('--model', type=str, default=None,
                       choices=['alexnet', 'alexnet_small', 'vgg16', 'vgg16_small', 
                               'resnet18', 'resnet34', 'efficientnet_small'],
                       help='Specific model to use')
    
    parser.add_argument('--activation', type=str, default=None,
                       choices=['relu', 'sigmoid', 'tanh', 'swish', 'gelu', 
                               'mish', 'elu', 'selu', 'custom_new'],
                       help='Specific activation function to use')
    
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with 1 epoch')
    
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU usage')
    
    args = parser.parse_args()
    
    # Setup
    print("="*80)
    print("ACTIVATION FUNCTION RESEARCH PROJECT")
    print("="*80)
    
    create_directories()
    
    if not args.no_gpu:
        setup_gpu()
    
    # Modify config for quick test
    if args.quick_test:
        print("\nRunning in QUICK TEST mode (1 epoch)")
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        config['epochs'] = 1
        config['datasets'] = ['mnist']
        config['models'] = ['alexnet_small']
        config['activations'] = ['relu', 'swish']
        
        # Save temporary config
        temp_config = 'temp_config.yaml'
        with open(temp_config, 'w') as f:
            yaml.dump(config, f)
        args.config = temp_config
    
    # Run experiments
    runner = ExperimentRunner(config_file=args.config)
    
    if args.dataset and args.model and args.activation:
        # Run single experiment
        print(f"\nRunning single experiment:")
        print(f"  Dataset: {args.dataset}")
        print(f"  Model: {args.model}")
        print(f"  Activation: {args.activation}\n")
        
        runner.run_single_experiment(args.dataset, args.model, args.activation)
    else:
        # Run all experiments
        print("\nRunning all experiments defined in config file\n")
        results_df = runner.run_all_experiments()
        
        print("\n" + "="*80)
        print("ALL EXPERIMENTS COMPLETED!")
        print("="*80)
        print(f"\nResults saved to: {runner.config['save_dir']}")
        print(f"Visualizations saved to: {runner.config['save_dir']}/plots")
        print(f"\nTotal experiments run: {len(results_df)}")
    
    # Cleanup temp config if created
    if args.quick_test and os.path.exists('temp_config.yaml'):
        os.remove('temp_config.yaml')
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()