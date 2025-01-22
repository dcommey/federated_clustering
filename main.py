# Add these at the top before other imports
import sys
import os

# First check dependencies
def check_dependencies():
    """Check all required dependencies."""
    required = {
        'torch': 'torch',
        'torchvision': 'torchvision',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn==1.3.2',  # Pin specific version
        'scipy': 'scipy==1.11.4'  # Pin specific version
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("Missing required packages. Please install:")
        print("pip install " + " ".join(missing))
        sys.exit(1)
    return True

check_dependencies()

# Now rest of imports
import logging
import argparse
import yaml
import json
import torch  # Add torch import here
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from src.federated import Server
from src.utils import setup_logging, save_checkpoint

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run federated learning with clustering')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    parser.add_argument('--experiment_name', type=str, default=None,
                      help='Name for the experiment run')
    parser.add_argument('--output_dir', type=str, default=None,
                      help='Output directory for results')
    parser.add_argument('--device', type=str, default=None, 
                      help='Device to run on (cpu/cuda)')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')
    return parser.parse_args()

def get_optimal_clustering_type(algorithm: str) -> str:
    """Get the optimal feature type for each clustering algorithm."""
    optimal_types = {
        'kmeans': 'model_weights',      # Good with high-dimensional, normalized data
        'hierarchical': 'label_distribution',  # Works well with distance-based features
        'spectral': 'data_features',    # Best with similarity-based features
        'gmm': 'model_weights'          # Good with continuous, normalized data
    }
    return optimal_types.get(algorithm, 'model_weights')

def setup_experiment_dir(config: Dict[str, Any], experiment_name: str = None, output_dir: str = None) -> str:
    """Create and setup experiment directory."""
    # Create results directory if it doesn't exist
    base_dir = Path(output_dir) if output_dir else Path("results")
    base_dir.mkdir(exist_ok=True)
    
    # Create timestamped experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = experiment_name or f"experiment_{timestamp}"
    exp_dir = base_dir / exp_name
    exp_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (exp_dir / "metrics").mkdir(exist_ok=True)
    (exp_dir / "models").mkdir(exist_ok=True)
    (exp_dir / "clustering").mkdir(exist_ok=True)
    
    # Save configuration
    with open(exp_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    return str(exp_dir)

def save_results(exp_dir: str, results: Dict[str, Any], server: Server):
    """Save experiment results and artifacts."""
    try:
        if not isinstance(server, Server):
            raise ValueError("Invalid server instance")
            
        # Create directories
        metrics_dir = Path(exp_dir) / "metrics"
        clustering_dir = Path(exp_dir) / "clustering"
        model_dir = Path(exp_dir) / "models"
        
        metrics_dir.mkdir(exist_ok=True)
        clustering_dir.mkdir(exist_ok=True)
        model_dir.mkdir(exist_ok=True)

        # Save metrics first
        metrics_path = metrics_dir / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        # Save clustering results
        if hasattr(server, 'clustering_manager'):
            clustering_metrics = server.clustering_manager.evaluate_clustering(
                server.clustering_manager.client_features
            )
            with open(clustering_dir / "clustering_metrics.json", 'w') as f:
                json.dump(clustering_metrics, f, indent=2)
            
        # Save model states last
        if hasattr(server, 'global_model'):
            model_path = model_dir / "final_model.pt"
            torch.save({
                'model_state': server.global_model.state_dict(),
                'training_config': server.config
            }, model_path)
            
            if server.best_model_state is not None:
                best_model_path = model_dir / "best_model.pt"
                torch.save({
                    'model_state': server.best_model_state,
                    'accuracy': server.best_accuracy,
                    'training_config': server.config
                }, best_model_path)
        
        logging.info(f"Results saved successfully in {exp_dir}")
        
    except Exception as e:
        logging.error(f"Error saving results: {e}")
        raise

def run_federated_learning(config_path: str, exp_dir: str) -> Dict[str, Any]:
    """Run federated learning experiment."""
    # Load and update configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update clustering configuration
    algo = config['clustering']['algorithm']
    config['clustering']['clustering_type'] = get_optimal_clustering_type(algo)
    
    # Initialize server
    server = Server(config_path)
    
    try:
        # Run training
        results = server.train()
        
        # Save results
        save_results(exp_dir, results, server)
        
        return results
        
    except Exception as e:
        logging.error(f"Error in federated learning: {e}")
        raise

def main():
    """Main execution function."""
    args = parse_args()
    
    try:
        # Load config and validate
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            
        required_keys = ['data', 'model', 'fed', 'clustering']
        missing_keys = [k for k in required_keys if k not in config]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")
            
        # Setup experiment directory
        exp_dir = setup_experiment_dir(config, args.experiment_name, args.output_dir)
        
        # Setup logging
        log_path = Path(exp_dir) / "experiment.log"
        setup_logging(str(log_path))
        
        logging.info(f"Starting experiment in directory: {exp_dir}")
        logging.info(f"Configuration: {config}")
        
        # Run experiment
        results = run_federated_learning(args.config, exp_dir)
        
        logging.info("Experiment completed successfully")
        if results.get('accuracy'):
            logging.info(f"Final Results - Accuracy: {results['accuracy'][-1]:.4f}")
        
    except Exception as e:
        logging.error(f"Experiment failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()