# src/utils.py

import torch
import logging
import os
import yaml
from typing import Dict, Any
import matplotlib.pyplot as plt
from datetime import datetime
import json
import numpy as np
from pathlib import Path

class NumpyEncoder(json.JSONEncoder):
    """Special JSON encoder for NumPy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def setup_logging(log_path: str) -> None:
    """Setup logging configuration."""
    # Create directory if it doesn't exist
    log_dir = os.path.dirname(log_path)
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

def get_device() -> torch.device:
    """Get available device (GPU or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(model: torch.nn.Module, path: str) -> None:
    """Save the model state ensuring all tensors are on CPU."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(state, path)

def load_model(model: torch.nn.Module, path: str) -> None:
    """Load the model state from file."""
    if os.path.exists(path):
        state = torch.load(path, map_location=get_device())
        model.load_state_dict(state)
    else:
        raise FileNotFoundError(f"No model file found at {path}")

def ensure_dir(path: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    Path(path).mkdir(parents=True, exist_ok=True)

def setup_experiment_dirs(base_dir: str) -> Dict[str, str]:
    """Setup all required experiment directories."""
    dirs = {
        'root': base_dir,
        'metrics': os.path.join(base_dir, 'metrics'),
        'models': os.path.join(base_dir, 'models'),
        'clustering': os.path.join(base_dir, 'clustering'),
        'logs': os.path.join(base_dir, 'logs')
    }
    
    for dir_path in dirs.values():
        ensure_dir(dir_path)
        
    return dirs

def save_results(exp_dir: str, results: Dict[str, Any], server: Any):
    """Save experiment results and artifacts."""
    try:
        # Create directories
        metrics_dir = Path(exp_dir) / "metrics"
        clustering_dir = Path(exp_dir) / "clustering"
        model_dir = Path(exp_dir) / "models"
        
        metrics_dir.mkdir(exist_ok=True)
        clustering_dir.mkdir(exist_ok=True)
        model_dir.mkdir(exist_ok=True)

        # Save metrics
        metrics_path = metrics_dir / "training_metrics.json"
        metrics_data = results if results is not None else {'error': 'No results available'}
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2, cls=NumpyEncoder)
            
        # Save clustering results if available
        if hasattr(server, 'clustering_manager') and server.clustering_manager is not None:
            # Save clustering metrics
            if hasattr(server.clustering_manager, 'evaluate_clustering'):
                clustering_metrics = server.clustering_manager.evaluate_clustering(
                    server.clustering_manager.client_features or {}
                )
                with open(clustering_dir / "clustering_metrics.json", 'w') as f:
                    json.dump(clustering_metrics, f, indent=2, cls=NumpyEncoder)
            
            # Save cluster assignments if available
            if (hasattr(server.clustering_manager, 'client_clusters') and 
                server.clustering_manager.client_clusters is not None):
                cluster_assignments = {
                    str(client_id): int(cluster_id)
                    for client_id, cluster_id in server.clustering_manager.client_clusters.items()
                }
                with open(clustering_dir / "cluster_assignments.json", 'w') as f:
                    json.dump(cluster_assignments, f, indent=2)
        
        # Save model states
        if hasattr(server, 'global_model'):
            # Save final model
            model_path = model_dir / "final_model.pt"
            torch.save({
                'model_state': server.global_model.state_dict(),
                'training_config': server.config
            }, model_path)
            
            # Save best model if available
            if hasattr(server, 'best_model_state') and server.best_model_state is not None:
                best_model_path = model_dir / "best_model.pt"
                torch.save({
                    'model_state': server.best_model_state,
                    'accuracy': getattr(server, 'best_accuracy', 0.0),
                    'training_config': server.config
                }, best_model_path)
            
        logging.info(f"Results saved successfully in {exp_dir}")
        
    except Exception as e:
        logging.error(f"Error saving results: {e}")
        raise

def save_checkpoint(model_state: Dict, path: str, config: Dict = None):
    """Safely save model checkpoint."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'model_state': model_state,
            'config': config
        }
        torch.save(checkpoint, path)
        return True
    except Exception as e:
        logging.error(f"Error saving checkpoint: {e}")
        return False