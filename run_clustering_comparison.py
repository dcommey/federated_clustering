#!/usr/bin/env python
# run_clustering_comparison.py

import yaml
import os
import json
from pathlib import Path
import itertools
from datetime import datetime
import logging
import subprocess
from typing import Dict, Any, List
import argparse
import sys

def generate_clustering_variants(base_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate different clustering configurations."""
    clustering_variants = {
        # Clustering algorithms and their specific configurations
        'kmeans': {
            'clustering_type': 'model_weights',
            'pca_components': [5, 10],
            'cluster_candidates': [[2, 3, 4], [3, 4, 5]]
        },
        'hierarchical': {
            'clustering_type': 'label_distribution',
            'pca_components': [5, 10],
            'cluster_candidates': [[2, 3, 4], [3, 4, 5]]
        },
        'spectral': {
            'clustering_type': 'data_features',
            'pca_components': [5, 10],
            'cluster_candidates': [[2, 3, 4], [3, 4, 5]]
        },
        'gmm': {
            'clustering_type': 'model_weights',
            'pca_components': [5, 10],
            'cluster_candidates': [[2, 3, 4], [3, 4, 5]]
        }
    }
    
    configs = []
    for algo, params in clustering_variants.items():
        for pca, candidates in itertools.product(
            params['pca_components'], 
            params['cluster_candidates']
        ):
            config = {
                'data': base_config['data'],
                'model': base_config['model'],
                'fed': base_config['fed'],
                'logging': base_config['logging'],
                'checkpointing': base_config['checkpointing'],
                'clustering': {
                    'algorithm': algo,
                    'clustering_type': params['clustering_type'],
                    'pca_components': pca,
                    'cluster_candidates': candidates,
                    'min_cluster_size': 2,
                    'drift_threshold': 0.3,
                    'update_frequency': 5
                }
            }
            configs.append(config)
    
    return configs

def setup_experiment_dir(base_dir: str = "clustering_experiments") -> str:
    """Setup experiment directory with timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path(base_dir) / f"comparison_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories with better organization
    (exp_dir / "results").mkdir(exist_ok=True)  # Each variant gets a subdir here
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "configs").mkdir(exist_ok=True)  # Create configs directory
    
    return str(exp_dir)

def run_experiment(config_path: str, exp_name: str, results_dir: str) -> bool:
    """Run a single experiment with organized results."""
    try:
        variant_dir = Path(results_dir) / exp_name
        variant_dir.mkdir(parents=True, exist_ok=True)
        
        python_path = sys.executable
        cmd = [
            python_path, 
            "main.py",
            "--config", str(config_path),
            "--experiment_name", exp_name,
            "--output_dir", str(variant_dir)
        ]
        
        # Run process and capture output
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            env=dict(os.environ, PYTHONPATH=str(Path.cwd()))  # Add current dir to PYTHONPATH
        )
        
        # Log output for debugging
        if result.stdout:
            logging.info(f"Experiment output:\n{result.stdout}")
        if result.stderr:
            logging.warning(f"Experiment warnings:\n{result.stderr}")
            
        return True
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Experiment failed: {str(e)}")
        if e.stdout:
            logging.error(f"stdout:\n{e.stdout}")
        if e.stderr:
            logging.error(f"stderr:\n{e.stderr}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return False

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load base configuration
    with open('config/config.yaml', 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Keep FL parameters constant
    base_config['fed'].update({
        'num_rounds': 10,
        'local_epochs': 5,
        'client_sample_ratio': 0.2,
        'patience': 5,
        'min_delta': 0.001
    })
    
    # Setup experiment directory
    exp_dir = setup_experiment_dir()
    logging.info(f"Starting clustering comparison experiments in: {exp_dir}")
    
    # Generate clustering variants
    variants = generate_clustering_variants(base_config)
    logging.info(f"Generated {len(variants)} clustering configurations")
    
    # Run experiments
    results = []
    for i, variant in enumerate(variants):
        variant_name = f"{variant['clustering']['algorithm']}_{i+1}"
        logging.info(f"\nRunning {variant_name} ({i+1}/{len(variants)})")
        logging.info(f"Configuration: {variant['clustering']}")
        
        # Merge with base config
        config = base_config.copy()
        config.update(variant)
        
        # Save configuration
        config_path = Path(exp_dir) / "configs" / f"{variant_name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Run experiment
        success = run_experiment(str(config_path), variant_name, str(Path(exp_dir) / "results"))
        
        # Record result
        result = {
            'variant': variant_name,
            'clustering_config': variant['clustering'],
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
        results.append(result)
        
        # Save intermediate results
        with open(Path(exp_dir) / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
    
    # Save final summary
    summary = {
        'total_experiments': len(variants),
        'successful_experiments': sum(1 for r in results if r['success']),
        'failed_experiments': sum(1 for r in results if not r['success']),
        'timestamp': datetime.now().isoformat(),
        'federated_learning_params': base_config['fed'],
        'results': results
    }
    
    with open(Path(exp_dir) / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info("\nExperiment Summary:")
    logging.info(f"Total experiments: {summary['total_experiments']}")
    logging.info(f"Successful experiments: {summary['successful_experiments']}")
    logging.info(f"Failed experiments: {summary['failed_experiments']}")
    logging.info(f"Results saved in: {exp_dir}")

def run_comparison():
    parser = argparse.ArgumentParser(description="Run main.py for multiple configs")
    parser.add_argument("--config_dir", type=str, default="clustering_experiments/configs",
                        help="Directory containing config files")
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    config_files = list(config_dir.glob("*.yaml"))

    if not config_files:
        print("No YAML config files found in the specified directory.")
        return

    for idx, config_file in enumerate(config_files, start=1):
        experiment_name = config_file.stem
        cmd = [
            "python",
            "main.py",
            "--config",
            str(config_file),
            "--experiment_name",
            experiment_name
        ]
        print(f"Running {experiment_name} ({idx}/{len(config_files)})")
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            print(f"Experiment failed for {experiment_name}: {e}")

if __name__ == "__main__":
    main()