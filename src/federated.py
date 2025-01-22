# src/federated.py

from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from tqdm import tqdm
import logging
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
from sklearn.preprocessing import StandardScaler

from .models import CNN
from .clustering import AdaptiveClusteringManager
from .data import DataManager
from .utils import get_device
from .metrics import MetricsCollector

class Client:
    """Enhanced Federated Learning Client."""
    
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        config_path: str,
        data_loader: torch.utils.data.DataLoader
    ):
        """Initialize FL client with enhanced capabilities."""
        self.client_id = client_id
        self.model = model
        self.device = get_device()
        self.model.to(self.device)
        
        # Load config
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.data_loader = data_loader
        
        # Enhanced optimizer configuration
        optimizer_name = self.config['model'].get('optimizer', 'sgd').lower()
        optimizer_params = {
            'lr': self.config['model']['learning_rate'],
            'weight_decay': self.config['model'].get('weight_decay', 0.0)
        }
        
        if optimizer_name == 'adam':
            optimizer_params.update({
                'betas': (self.config['model'].get('adam_beta1', 0.9),
                         self.config['model'].get('adam_beta2', 0.999)),
                'eps': self.config['model'].get('adam_epsilon', 1e-8)
            })
            self.optimizer = optim.Adam(self.model.parameters(), **optimizer_params)
        else:  # SGD
            optimizer_params['momentum'] = self.config['model'].get('momentum', 0.9)
            self.optimizer = optim.SGD(self.model.parameters(), **optimizer_params)
        
        # Loss function with optional label smoothing
        smoothing = self.config['model'].get('label_smoothing', 0.0)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
        
        # Training history
        self.training_history = defaultdict(list)
        
    def train(self, epochs: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Train the model with enhanced monitoring and stability."""
        data_count = len(self.data_loader.dataset)
        if data_count == 0:
            logging.warning(f"Client {self.client_id} has no data. Skipping training.")
            return self.model.state_dict(), {'loss': 0.0, 'accuracy': 0.0}

        self.model.train()
        metrics = defaultdict(float)
        total_samples = 0
        
        # Initialize gradient clipping
        max_grad_norm = self.config['model'].get('max_grad_norm', None)
        
        for epoch in range(epochs):
            epoch_metrics = defaultdict(float)
            epoch_samples = 0
            
            for batch_idx, (data, target) in enumerate(self.data_loader):
                try:
                    batch_metrics = self._train_batch(data, target, max_grad_norm)
                    batch_size = data.size(0)
                    
                    # Update epoch metrics
                    for key, value in batch_metrics.items():
                        epoch_metrics[key] += value * batch_size
                    epoch_samples += batch_size
                    
                except Exception as e:
                    logging.error(f"Error in batch training: {str(e)}")
                    continue
            
            # Compute epoch averages
            if epoch_samples > 0:
                for key in epoch_metrics:
                    avg_value = epoch_metrics[key] / epoch_samples
                    metrics[key] += avg_value
                    self.training_history[f'epoch_{key}'].append(float(avg_value))
                total_samples += epoch_samples
        
        # Compute final averages
        if total_samples > 0:
            metrics = {k: float(v / epochs) for k, v in metrics.items()}
        else:
            metrics = {'loss': float('inf'), 'accuracy': 0.0}
        
        return self.model.state_dict(), metrics
    
    def _train_batch(self, data: torch.Tensor, target: torch.Tensor, 
                    max_grad_norm: Optional[float]) -> Dict[str, float]:
        """Train a single batch with enhanced error handling."""
        try:
            # Move data to device and ensure correct types
            data = data.float().to(self.device)
            target = target.long().to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass with gradient clipping
            loss.backward()
            if max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.optimizer.step()
            
            # Compute metrics
            with torch.no_grad():
                preds = output.argmax(dim=1)
                correct = (preds == target).sum().item()
                metrics = {
                    'loss': loss.item(),
                    'accuracy': correct / data.size(0)
                }
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error in batch training: {str(e)}")
            return {'loss': float('inf'), 'accuracy': 0.0}
        
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model with enhanced metrics."""
        self.model.eval()
        metrics = defaultdict(float)
        total_samples = 0
        
        try:
            with torch.no_grad():
                for data, target in self.data_loader:
                    # Move data to device
                    data = data.float().to(self.device)
                    target = target.long().to(self.device)
                    
                    # Forward pass
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    
                    # Compute metrics
                    batch_size = data.size(0)
                    preds = output.argmax(dim=1)
                    correct = (preds == target).sum().item()
                    
                    # Update metrics
                    metrics['loss'] += loss.item() * batch_size
                    metrics['accuracy'] += correct
                    total_samples += batch_size
            
            # Compute averages
            if total_samples > 0:
                metrics = {k: float(v / total_samples) for k, v in metrics.items()}
            else:
                metrics = {'loss': float('inf'), 'accuracy': 0.0}
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error in evaluation: {str(e)}")
            return {'loss': float('inf'), 'accuracy': 0.0}

    def update_weights(self, weights: Dict[str, torch.Tensor]):
        """Update local model weights with error checking."""
        try:
            # Validate weights before updating
            if not all(name in self.model.state_dict() for name in weights):
                raise ValueError("Received weights don't match model architecture")
            
            # Ensure weights are on the correct device
            weights = {k: v.to(self.device) for k, v in weights.items()}
            self.model.load_state_dict(weights)
            
        except Exception as e:
            logging.error(f"Error updating weights for client {self.client_id}: {str(e)}")

    def get_data_features(self) -> Optional[np.ndarray]:
        """Extract features from client's data."""
        try:
            features = []
            for data, _ in self.data_loader:
                batch_features = data.numpy().reshape(data.size(0), -1)
                features.append(batch_features)
            
            if not features:
                return None
                
            return np.vstack(features).mean(axis=0)
            
        except Exception as e:
            logging.error(f"Error extracting data features: {str(e)}")
            return None

    def get_label_distribution(self) -> Optional[np.ndarray]:
        """Get label distribution with error handling."""
        try:
            labels = []
            for _, target in self.data_loader:
                labels.extend(target.numpy())
                
            if not labels:
                return None
                
            labels = np.array(labels)
            num_classes = self.config['model']['num_classes']
            
            # Compute distribution with smoothing
            counts = np.bincount(labels, minlength=num_classes)
            smoothing = 0.1  # Laplace smoothing
            smoothed_dist = (counts + smoothing) / (counts.sum() + smoothing * num_classes)
            
            return smoothed_dist
            
        except Exception as e:
            logging.error(f"Error computing label distribution: {str(e)}")
            return None

class Server:
    """Enhanced Federated Learning Server."""
    
    def __init__(self, config_path: str):
        """Initialize FL server with enhanced capabilities."""
        self.device = get_device()
        self.config_path = config_path
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Initialize components
        self.data_manager = DataManager(config_path)
        self.data_manager.load_dataset()
        self.data_manager.create_client_data()
        
        # Initialize global model
        self.global_model = self._initialize_model()
        
        # Initialize clustering manager
        self.clustering_manager = AdaptiveClusteringManager(config_path)
        
        # Initialize clients
        self.clients = self._initialize_clients()
        
        # Initialize client features
        self._initialize_client_features()
        
        # Metrics storage
        self.metrics_collector = MetricsCollector()
        self.best_model_state = None
        self.best_accuracy = 0.0
        
        # Load checkpoint if exists
        self._load_checkpoint()
    
    def _initialize_model(self) -> nn.Module:
        """Initialize global model with configuration."""
        try:
            model = CNN(
                input_channels=self.config['model']['input_channels'],
                num_classes=self.config['model']['num_classes']
            ).to(self.device)
            
            # Initialize weights if specified
            init_method = self.config['model'].get('weight_init', 'kaiming')
            if init_method == 'kaiming':
                for m in model.modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        nn.init.kaiming_normal_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
            
            return model
            
        except Exception as e:
            logging.error(f"Error initializing model: {str(e)}")
            raise
    
    def _initialize_clients(self) -> Dict[int, Client]:
        """Initialize clients with error handling."""
        clients = {}
        for client_id in range(self.config['data']['num_clients']):
            try:
                client_model = copy.deepcopy(self.global_model)
                client_loader = self.data_manager.get_client_loader(client_id)
                clients[client_id] = Client(
                    client_id=client_id,
                    model=client_model,
                    config_path=self.config_path,
                    data_loader=client_loader
                )
            except Exception as e:
                logging.error(f"Error initializing client {client_id}: {str(e)}")
                continue
        return clients
    
    def _initialize_client_features(self):
        """Initialize client features for clustering."""
        for client_id, client in self.clients.items():
            try:
                features = {}
                
                if self.clustering_manager.algorithm.clustering_type in ['model_weights', 'hybrid']:
                    features['model_weights'] = client.model.state_dict()
                
                if self.clustering_manager.algorithm.clustering_type in ['label_distribution', 'hybrid']:
                    features['label_distribution'] = client.get_label_distribution()
                
                if self.clustering_manager.algorithm.clustering_type in ['data_features', 'hybrid']:
                    features['data_features'] = client.get_data_features()
                
                self.clustering_manager.update_client_features(
                    client_id=client_id,
                    **features
                )
                
            except Exception as e:
                logging.error(f"Error initializing features for client {client_id}: {str(e)}")
                continue
    
    def train_round(self) -> Dict[str, float]:
        """Conduct one round of federated training."""
        # Select clients
        selected_clients = self._select_clients()
        
        # Train selected clients
        updates = self._train_selected_clients(selected_clients)
        
        # Aggregate updates
        if updates['successful']:
            self._aggregate_updates(updates)
        
        # Evaluate and update metrics
        round_metrics = self._evaluate_round(updates)
        
        # Update clustering if needed
        if self.clustering_manager.should_cluster():
            self._update_clustering()
        
        # Save checkpoint
        self._save_checkpoint()
        
        return round_metrics
    
    def _select_clients(self) -> List[int]:
        """Select clients for training round."""
        num_clients = max(
            2,
            int(self.config['fed']['client_sample_ratio'] * len(self.clients))
        )
        
        if self.clustering_manager.client_clusters:
            return self.clustering_manager._select_clients_stratified(num_clients)
        else:
            return list(np.random.choice(list(self.clients.keys()), 
                                       num_clients, replace=False))
    
    def _train_selected_clients(self, selected_clients: List[int]) -> Dict:
        """Train selected clients with enhanced monitoring."""
        updates = {
            'successful': {},
            'failed': [],
            'metrics': defaultdict(list)
        }
        
        for client_id in tqdm(selected_clients, desc="Training clients"):
            try:
                client = self.clients[client_id]
                weights, metrics = client.train(self.config['fed']['local_epochs'])
                
                # Validate updates
                if self._validate_update(weights, metrics):
                    updates['successful'][client_id] = weights
                    for k, v in metrics.items():
                        updates['metrics'][k].append(float(v))
                else:
                    updates['failed'].append(client_id)
                    
            except Exception as e:
                logging.error(f"Error training client {client_id}: {str(e)}")
                updates['failed'].append(client_id)
        
        return updates
    
    def _validate_update(self, weights: Dict[str, torch.Tensor], 
                        metrics: Dict[str, float]) -> bool:
        """Validate client updates."""
        try:
            # Check if weights match model architecture
            if not all(name in self.global_model.state_dict() for name in weights):
                return False
            
            # Check for NaN or inf values
            if any(torch.isnan(w).any() or torch.isinf(w).any() for w in weights.values()):
                return False
            
            # Check metrics are valid
            if any(not isinstance(v, (int, float)) or np.isnan(v) or np.isinf(v) 
                   for v in metrics.values()):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _aggregate_updates(self, updates: Dict) -> None:
        """Aggregate client updates with cluster-aware weighting."""
        if not updates['successful']:
            return
        
        try:
            if self.clustering_manager.client_clusters:
                # Cluster-based aggregation
                cluster_weights = self._aggregate_cluster_weights(updates['successful'])
                global_weights = self._aggregate_global_weights(cluster_weights)
            else:
                # Standard FedAvg
                global_weights = self._aggregate_weights(updates['successful'])
            
            # Update global model
            self.global_model.load_state_dict(global_weights)
            
            # Update all clients
            for client in self.clients.values():
                client.update_weights(global_weights)
                
        except Exception as e:
            logging.error(f"Error in update aggregation: {str(e)}")
    
    def _aggregate_weights(self, weights_dict: Dict[int, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Aggregate weights using FedAvg algorithm."""
        averaged_weights = {}
        
        try:
            for param_name in weights_dict[list(weights_dict.keys())[0]].keys():
                # Stack parameters and ensure they're float type
                stacked = torch.stack([
                    weights[param_name].to(dtype=torch.float32, device=self.device)
                    for weights in weights_dict.values()
                ])
                
                # Compute mean
                averaged_weights[param_name] = stacked.mean(dim=0)
                
        except Exception as e:
            logging.error(f"Error in weight averaging: {str(e)}")
            # Fallback to first client's weights
            first_client = list(weights_dict.keys())[0]
            averaged_weights = weights_dict[first_client]
            
        return averaged_weights
    
    def _aggregate_cluster_weights(
        self,
        client_weights: Dict[int, Dict[str, torch.Tensor]]
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Aggregate weights within each cluster."""
        cluster_weights = {}
        
        # Group clients by cluster
        cluster_groups = defaultdict(dict)
        for client_id, weights in client_weights.items():
            cluster_id = self.clustering_manager.get_client_cluster(client_id)
            if cluster_id is not None:
                cluster_groups[cluster_id][client_id] = weights
        
        # Aggregate within clusters
        for cluster_id, group_weights in cluster_groups.items():
            cluster_weights[cluster_id] = self._aggregate_weights(group_weights)
            
        return cluster_weights
    
    def _aggregate_global_weights(
        self,
        cluster_weights: Dict[int, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Aggregate weights across clusters."""
        if not cluster_weights:
            return self.global_model.state_dict()
        
        # Get cluster sizes for weighted averaging
        cluster_sizes = defaultdict(int)
        for client_id, cluster_id in self.clustering_manager.client_clusters.items():
            cluster_sizes[cluster_id] += 1
        
        total_size = sum(cluster_sizes.values())
        
        # Compute weighted average
        global_weights = {}
        for param_name in cluster_weights[list(cluster_weights.keys())[0]].keys():
            weighted_sum = torch.zeros_like(
                cluster_weights[list(cluster_weights.keys())[0]][param_name]
            )
            
            for cluster_id, weights in cluster_weights.items():
                weight = cluster_sizes[cluster_id] / total_size
                weighted_sum += weights[param_name] * weight
            
            global_weights[param_name] = weighted_sum
            
        return global_weights
    
    def _evaluate_round(self, updates: Dict) -> Dict[str, float]:
        """Evaluate round performance."""
        metrics = {
            'num_successful': len(updates['successful']),
            'num_failed': len(updates['failed']),
        }
        
        # Add client metrics
        for metric_name, values in updates['metrics'].items():
            if values:
                metrics[f'client_{metric_name}_mean'] = float(np.mean(values))
                metrics[f'client_{metric_name}_std'] = float(np.std(values))
        
        # Evaluate global model
        global_metrics = self.evaluate_global()
        metrics.update(global_metrics)
        
        # Update best model if needed
        if global_metrics['accuracy'] > self.best_accuracy:
            self.best_accuracy = global_metrics['accuracy']
            self.best_model_state = copy.deepcopy(self.global_model.state_dict())
        
        # Add to metrics collector
        self.metrics_collector.add_round_metrics(metrics)
        
        return metrics
    
    def _update_clustering(self) -> None:
        """Update clustering with current client features."""
        try:
            self._initialize_client_features()  # Update features
            cluster_groups = self.clustering_manager.perform_clustering()
            
            logging.info(f"Updated clustering: {len(cluster_groups)} clusters")
            
        except Exception as e:
            logging.error(f"Error updating clustering: {str(e)}")
    
    def evaluate_global(self) -> Dict[str, float]:
        """Evaluate global model performance."""
        self.global_model.eval()
        metrics = defaultdict(float)
        total_samples = 0
        
        try:
            test_loader = self.data_manager.get_test_loader()
            criterion = nn.CrossEntropyLoss()
            
            with torch.no_grad():
                for data, target in test_loader:
                    data = data.float().to(self.device)
                    target = target.long().to(self.device)
                    
                    output = self.global_model(data)
                    loss = criterion(output, target)
                    
                    batch_size = data.size(0)
                    preds = output.argmax(dim=1)
                    correct = (preds == target).sum().item()
                    
                    metrics['loss'] += loss.item() * batch_size
                    metrics['accuracy'] += correct
                    total_samples += batch_size
            
            # Compute averages
            if total_samples > 0:
                metrics = {k: float(v / total_samples) for k, v in metrics.items()}
            else:
                metrics = {'loss': float('inf'), 'accuracy': 0.0}
            
        except Exception as e:
            logging.error(f"Error in global evaluation: {str(e)}")
            metrics = {'loss': float('inf'), 'accuracy': 0.0}
        
        return metrics
    
    def train(self) -> Dict[str, List[float]]:
        """Complete training process."""
        try:
            # Initial clustering
            self._update_clustering()
            
            # Training loops
            for round_num in range(self.config['fed']['num_rounds']):
                logging.info(f"Starting round {round_num + 1}")
                
                # Train round
                round_metrics = self.train_round()
                
                # Log progress
                logging.info(
                    f"Round {round_num + 1} completed: "
                    f"Accuracy = {round_metrics['accuracy']:.4f}, "
                    f"Loss = {round_metrics['loss']:.4f}"
                )
                
                # Early stopping check
                if self._check_early_stopping():
                    logging.info("Early stopping triggered")
                    break
            
            # Restore best model
            if self.best_model_state is not None:
                self.global_model.load_state_dict(self.best_model_state)
            
            return self.metrics_collector.get_metrics_history()
            
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            return {'accuracy': [0.0], 'loss': [float('inf')]}
    
    def _check_early_stopping(self) -> bool:
        """Enhanced early stopping with minimum rounds requirement."""
        patience = self.config['fed'].get('patience', 10)
        min_delta = self.config['fed'].get('min_delta', 0.0005)
        min_rounds = self.config['fed'].get('min_rounds_before_early_stopping', 5)
        
        # Don't stop before minimum rounds
        if self.clustering_manager.round_number < min_rounds:
            return False
        
        metrics_history = self.metrics_collector.get_metrics_history()
        accuracies = metrics_history.get('accuracy', [])
        
        if len(accuracies) < patience + 1:
            return False
        
        # Check if accuracy hasn't improved
        best_accuracy = max(accuracies[:-patience])
        recent_best = max(accuracies[-patience:])
        
        return recent_best - best_accuracy < min_delta
    
    def _save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_dir = self.config['checkpointing'].get('dir', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'model_state': self.global_model.state_dict(),
            'best_model_state': self.best_model_state,
            'best_accuracy': self.best_accuracy,
            'round_number': self.clustering_manager.round_number,
            'metrics_history': self.metrics_collector.get_metrics_history()
        }
        
        path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, path)
    
    def _load_checkpoint(self):
        """Load training checkpoint."""
        checkpoint_dir = self.config['checkpointing'].get('dir', 'checkpoints')
        path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
        
        if not os.path.exists(path):
            return
        
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            
            self.global_model.load_state_dict(checkpoint['model_state'])
            self.best_model_state = checkpoint['best_model_state']
            self.best_accuracy = checkpoint['best_accuracy']
            self.clustering_manager.round_number = checkpoint['round_number']
            
            # Restore metrics history
            for metric_name, values in checkpoint['metrics_history'].items():
                for value in values:
                    self.metrics_collector.add_metric(metric_name, value)
                    
        except Exception as e:
            logging.error(f"Error loading checkpoint: {str(e)}")