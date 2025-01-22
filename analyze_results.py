import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List, Any
import logging

def load_experiment_results(exp_dir: Path) -> Dict[str, Any]:
    """Load results from a single experiment directory."""
    try:
        # Load training metrics
        metrics_path = exp_dir / "metrics" / "training_metrics.json"
        with open(metrics_path) as f:
            training_metrics = json.load(f)
        
        # Load clustering metrics
        clustering_path = exp_dir / "clustering" / "clustering_metrics.json"
        with open(clustering_path) as f:
            clustering_metrics = json.load(f)
            
        # Load config
        config_path = exp_dir / "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        return {
            'exp_name': exp_dir.name,
            'training_metrics': training_metrics,
            'clustering_metrics': clustering_metrics,
            'config': config
        }
    except Exception as e:
        logging.error(f"Error loading results from {exp_dir}: {e}")
        return None

def aggregate_results(base_dir: str = "clustering_experiments") -> pd.DataFrame:
    """Aggregate results from all experiments into a DataFrame."""
    results = []
    base_path = Path(base_dir)
    
    for exp_dir in base_path.glob("comparison_*"):
        # Get experiment timestamp
        timestamp = exp_dir.name.split('_')[1]
        
        # Load results for each variant
        for variant_dir in (exp_dir / "results").glob("*"):
            exp_results = load_experiment_results(variant_dir)
            if exp_results is None:
                continue
                
            # Extract key metrics
            final_metrics = {
                'timestamp': timestamp,
                'experiment': exp_results['exp_name'],
                'algorithm': exp_results['config']['clustering']['algorithm'],
                'clustering_type': exp_results['config']['clustering']['clustering_type'],
                'pca_components': exp_results['config']['clustering']['pca_components'],
                'final_accuracy': exp_results['training_metrics']['accuracy'][-1],
                'best_accuracy': max(exp_results['training_metrics']['accuracy']),
                'convergence_round': len(exp_results['training_metrics']['accuracy']),
                'silhouette_score': exp_results['clustering_metrics'].get('silhouette_score', 0),
                'stability_score': exp_results['clustering_metrics'].get('stability_score', 0)
            }
            results.append(final_metrics)
    
    return pd.DataFrame(results)

def plot_algorithm_comparison(df: pd.DataFrame, save_dir: str = "analysis"):
    """Create comparative visualizations."""
    Path(save_dir).mkdir(exist_ok=True)
    
    # Accuracy comparison
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='algorithm', y='final_accuracy')
    plt.title('Accuracy Distribution by Clustering Algorithm')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/accuracy_comparison.png")
    plt.close()
    
    # Clustering quality metrics
    plt.figure(figsize=(12, 6))
    metrics = ['silhouette_score', 'stability_score']
    for i, metric in enumerate(metrics):
        plt.subplot(1, 2, i+1)
        sns.boxplot(data=df, x='algorithm', y=metric)
        plt.title(f'{metric.replace("_", " ").title()} by Algorithm')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/clustering_metrics.png")
    plt.close()
    
    # Convergence analysis
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='convergence_round', y='final_accuracy', 
                   hue='algorithm', style='clustering_type')
    plt.title('Convergence vs Accuracy')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/convergence_analysis.png")
    plt.close()

def generate_summary_report(df: pd.DataFrame, save_dir: str = "analysis"):
    """Generate a detailed summary report."""
    report = {
        'best_configuration': df.loc[df['final_accuracy'].idxmax()].to_dict(),
        'algorithm_performance': df.groupby('algorithm').agg({
            'final_accuracy': ['mean', 'std', 'max'],
            'convergence_round': 'mean',
            'silhouette_score': 'mean'
        }).to_dict(),
        'clustering_type_performance': df.groupby('clustering_type').agg({
            'final_accuracy': ['mean', 'std', 'max']
        }).to_dict()
    }
    
    with open(f"{save_dir}/summary_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

def main():
    """Analyze and compare all experiment results."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create analysis directory
    analysis_dir = "analysis"
    Path(analysis_dir).mkdir(exist_ok=True)
    
    # Aggregate results
    df = aggregate_results()
    
    # Generate visualizations
    plot_algorithm_comparison(df, analysis_dir)
    
    # Generate summary report
    report = generate_summary_report(df, analysis_dir)
    
    # Print key findings
    print("\nKey Findings:")
    print(f"Best Configuration: {report['best_configuration']['algorithm']} "
          f"with {report['best_configuration']['clustering_type']}")
    print(f"Best Accuracy: {report['best_configuration']['final_accuracy']:.4f}")
    
    # Save full DataFrame
    df.to_csv(f"{analysis_dir}/full_results.csv", index=False)

if __name__ == "__main__":
    main()
