# src/metrics.py

import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score, 
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
import torch
import logging
from collections import defaultdict
import json
import os
from datetime import datetime

class MetricsCollector:
    """Enhanced metrics collection for federated learning and clustering."""
    
    def __init__(self):
        """Initialize metrics storage with comprehensive tracking."""
        # Core metrics storage
        self.metrics = defaultdict(list)
        self.round_metrics = {}
        
        # Clustering specific metrics
        self.clustering_metrics = defaultdict(list)
        self.stability_history = []
        self.cluster_assignments = []
        
        # Client performance tracking
        self.client_metrics = defaultdict(lambda: defaultdict(list))
        
        # Best performance tracking
        self.best_metrics = {}
        self.best_round = None
        
        # Initialize statistics
        self._initialize_statistics()
        
        # Initialize history
        self.history = defaultdict(list)
    
    def add_metric(self, metric_name: str, value: float):
        """Add a single metric value."""
        try:
            self.history[metric_name].append(float(value))
        except Exception as e:
            logging.error(f"Error adding metric {metric_name}: {e}")
    
    def _initialize_statistics(self):
        """Initialize statistical tracking."""
        self.statistics = {
            'running_mean': defaultdict(float),
            'running_std': defaultdict(float),
            'running_count': defaultdict(int),
            'peaks': defaultdict(list),
            'troughs': defaultdict(list)
        }
    
    def add_round_metrics(self, metrics: Dict[str, float], round_num: Optional[int] = None):
        """Add metrics for current round."""
        try:
            # Process metrics
            processed_metrics = {}
            for name, value in metrics.items():
                try:
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    processed_value = float(value)
                    
                    if np.isnan(processed_value) or np.isinf(processed_value):
                        logging.warning(f"Invalid value for metric {name}: {value}")
                        continue
                    
                    processed_metrics[name] = processed_value
                    self.metrics[name].append(processed_value)
                    self.history[name].append(processed_value)
                    
                except (TypeError, ValueError) as e:
                    logging.warning(f"Could not convert metric {name}: {e}")
                    continue
            
            # Store round metrics
            if round_num is not None:
                self.metrics['round'].append(round_num)
            
            self.round_metrics = processed_metrics
            
        except Exception as e:
            logging.error(f"Error adding round metrics: {e}")
            self.round_metrics = {'error': str(e)}
    
    def add_client_metrics(self, client_id: int, metrics: Dict[str, float]):
        """Track per-client performance metrics."""
        try:
            for name, value in metrics.items():
                if isinstance(value, (int, float, np.number)) and not np.isnan(value):
                    self.client_metrics[client_id][name].append(float(value))
        except Exception as e:
            logging.error(f"Error adding client metrics for client {client_id}: {e}")

    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Generate comprehensive metrics summary."""
        summary = {}
        
        try:
            # Process each metric
            for metric_name, values in self.metrics.items():
                if not values:
                    continue
                
                metric_summary = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'last': float(values[-1])
                }
                
                # Add trend analysis
                metric_summary.update(self._compute_trend_metrics(values))
                
                summary[metric_name] = metric_summary
            
            return summary
            
        except Exception as e:
            logging.error(f"Error generating metrics summary: {e}")
            return {}
    
    def _compute_trend_metrics(self, values: List[float], window: int = 5) -> Dict[str, Any]:
        """Compute trend metrics for a sequence of values."""
        if len(values) < 2:
            return {'trend': 'stable', 'trend_strength': 0.0}
        
        try:
            # Compute recent trend
            recent_values = values[-min(window, len(values)):]
            slope, _ = np.polyfit(range(len(recent_values)), recent_values, 1)
            
            # Determine trend direction and strength
            trend_strength = abs(slope) / (np.std(recent_values) + 1e-10)
            
            if trend_strength < 0.1:
                trend = 'stable'
            elif slope > 0:
                trend = 'increasing'
            else:
                trend = 'decreasing'
            
            return {
                'trend': trend,
                'trend_strength': float(trend_strength)
            }
            
        except Exception as e:
            logging.error(f"Error computing trend metrics: {e}")
            return {'trend': 'stable', 'trend_strength': 0.0}
    
    def save_metrics(self, save_dir: str):
        """Save metrics to file."""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            # Save metrics
            metrics_path = os.path.join(save_dir, f'metrics_{timestamp}.json')
            with open(metrics_path, 'w') as f:
                json.dump({
                    'metrics': self.metrics,
                    'clustering_metrics': self.clustering_metrics,
                    'best_metrics': self.best_metrics,
                    'best_round': self.best_round
                }, f, indent=2)
            
            # Save summary
            summary_path = os.path.join(save_dir, f'summary_{timestamp}.json')
            with open(summary_path, 'w') as f:
                json.dump(self.get_metrics_summary(), f, indent=2)
                
        except Exception as e:
            logging.error(f"Error saving metrics: {e}")
    
    def load_metrics(self, metrics_path: str):
        """Load metrics from disk."""
        try:
            with open(metrics_path, 'r') as f:
                data = json.load(f)
            
            self.metrics = defaultdict(list, data.get('metrics', {}))
            self.clustering_metrics = defaultdict(list, data.get('clustering_metrics', {}))
            self.best_metrics = data.get('best_metrics', {})
            self.best_round = data.get('best_round', None)
            
            # Reinitialize statistics
            self._initialize_statistics()
            
            # Recompute statistics
            for metric_name, values in self.metrics.items():
                for value in values:
                    self._update_statistics(metric_name, value)
                    
        except Exception as e:
            logging.error(f"Error loading metrics from {metrics_path}: {e}")

    def _update_statistics(self, metric_name: str, value: float):
        """Update running statistics for a metric."""
        try:
            n = self.statistics['running_count'][metric_name]
            old_mean = self.statistics['running_mean'][metric_name]
            
            # Update running count
            n += 1
            self.statistics['running_count'][metric_name] = n
            
            # Update running mean
            delta = value - old_mean
            new_mean = old_mean + delta / n
            self.statistics['running_mean'][metric_name] = new_mean
            
            # Update running standard deviation using Welford's method
            if n > 1:
                delta2 = value - new_mean
                m2 = self.statistics['running_std'][metric_name] * (n - 2) + delta * delta2
                self.statistics['running_std'][metric_name] = m2 / (n - 1)
            
            # Track peaks and troughs
            if n > 1:
                metrics_list = self.metrics[metric_name]
                if len(metrics_list) >= 3:
                    if metrics_list[-2] > metrics_list[-3] and metrics_list[-2] > value:
                        self.statistics['peaks'][metric_name].append(
                            (len(metrics_list)-2, metrics_list[-2])
                        )
                    elif metrics_list[-2] < metrics_list[-3] and metrics_list[-2] < value:
                        self.statistics['troughs'][metric_name].append(
                            (len(metrics_list)-2, metrics_list[-2])
                        )
        except Exception as e:
            logging.error(f"Error updating statistics for {metric_name}: {e}")

    def get_metrics_history(self) -> Dict[str, List[float]]:
        """Retrieve metrics history."""
        return dict(self.history)