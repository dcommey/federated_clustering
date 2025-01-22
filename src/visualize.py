import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any

def plot_clustering_comparison(results: Dict[str, Dict[str, Any]], save_path: str = None):
    """Plot clustering comparison results."""
    metrics = ['silhouette_score', 'davies_bouldin_score', 'calinski_harabasz_score', 'accuracy']
    algorithms = list(results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, metric in enumerate(metrics):
        values = []
        labels = []
        
        for algo in algorithms:
            if 'error' not in results[algo]:
                if metric == 'accuracy':
                    value = results[algo].get('accuracy', 0)
                else:
                    value = results[algo].get('clustering_metrics', {}).get(metric, 0)
                values.append(value)
                labels.append(f"{algo}\n({results[algo]['feature_type']})")
        
        axes[idx].bar(labels, values)
        axes[idx].set_title(metric.replace('_', ' ').title())
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
