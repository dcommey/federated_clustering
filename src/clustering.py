import numpy as np
import torch
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score, 
    calinski_harabasz_score,
    adjusted_rand_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from scipy.stats import wasserstein_distance
from typing import Dict, List, Tuple, Any, Optional
from abc import ABC, abstractmethod
import logging
import yaml
import os

class BaseClusteringAlgorithm(ABC):
    """Base class for all clustering algorithms."""
    
    CLUSTERING_TYPES = ['model_weights', 'label_distribution', 'data_features', 'hybrid']
    
    def __init__(self, config_path: str):
        """Initialize clustering algorithm with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Initialize basic parameters
        self.min_clusters = self.config['clustering'].get('min_clusters', 2)
        self.max_clusters = self.config['clustering'].get('max_clusters', 10)
        self.n_clusters = self.min_clusters  # Initial value
        self.clustering_type = self.config['clustering'].get('clustering_type', 'model_weights')
        
        # PCA settings
        self.pca_components = self.config['clustering'].get('pca_components', 5)
        
        # Feature normalization
        self.feature_scaler = StandardScaler()
        
        # Store config path for later use
        self.config_path = config_path
        
    @abstractmethod
    def cluster_clients(self, client_features: np.ndarray) -> np.ndarray:
        """Cluster clients based on their features."""
        pass

    def extract_features(
        self,
        model_weights: Optional[Dict[str, torch.Tensor]] = None,
        data_features: Optional[np.ndarray] = None,
        label_dist: Optional[np.ndarray] = None,
        feature_importance: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """Extract and combine features based on importance weights."""
        features = []
        importance = feature_importance or {'model_weights': 1.0, 'label_distribution': 1.0, 'data_features': 1.0}
        
        try:
            if self.clustering_type == 'model_weights' and model_weights is not None:
                model_feat = self._extract_model_features(model_weights)
                features.append(model_feat * importance.get('model_weights', 1.0))
                
            elif self.clustering_type == 'label_distribution' and label_dist is not None:
                # Ensure label_dist is 1D
                label_feat = label_dist.ravel()
                features.append(label_feat * importance.get('label_distribution', 1.0))
                
            elif self.clustering_type == 'data_features' and data_features is not None:
                # Ensure data_features is 1D
                data_feat = data_features.ravel()
                features.append(data_feat * importance.get('data_features', 1.0))
                
            elif self.clustering_type == 'hybrid':
                if model_weights is not None:
                    model_feat = self._extract_model_features(model_weights)
                    features.append(model_feat * importance.get('model_weights', 1.0))
                if label_dist is not None:
                    label_feat = label_dist.ravel()
                    features.append(label_feat * importance.get('label_distribution', 1.0))
                if data_features is not None:
                    data_feat = data_features.ravel()
                    features.append(data_feat * importance.get('data_features', 1.0))
            
            if not features:
                raise ValueError(f"No valid features available for clustering type: {self.clustering_type}")
            
            # Combine features
            combined = np.concatenate(features).reshape(1, -1)
            num_samples, num_features = combined.shape

            # Skip or adjust PCA if fewer than 2 samples
            if num_samples < 2:
                return combined.flatten()[:self.pca_components]
            
            safe_components = min(self.pca_components, num_samples, num_features)
            if safe_components > 1:
                pca = PCA(n_components=safe_components)
                combined = pca.fit_transform(combined).ravel()
            elif combined.size < self.pca_components:
                # Pad with zeros if needed
                padding = np.zeros(self.pca_components - combined.size)
                combined = np.concatenate([combined, padding])
            
            return combined
            
        except Exception as e:
            logging.error(f"Error in feature extraction: {str(e)}")
            return np.zeros(self.pca_components)

    def _extract_model_features(self, model_weights: Dict[str, torch.Tensor]) -> np.ndarray:
        """Extract and normalize features from model weights."""
        try:
            features = []
            for param_name, weight in model_weights.items():
                # Convert to numpy and flatten
                weight_np = weight.cpu().detach().numpy()
                features.append(weight_np.reshape(1, -1))
            
            # Concatenate all features
            if not features:
                return np.zeros(self.pca_components)
                
            combined = np.concatenate(features, axis=1).reshape(1, -1)
            num_samples, num_features = combined.shape

            if num_samples < 2:
                return np.zeros(self.pca_components)
            
            safe_components = min(self.pca_components, num_samples, num_features)
            if safe_components > 1:
                pca = PCA(n_components=safe_components)
                combined = pca.fit_transform(combined)
            elif combined.shape[1] < self.pca_components:
                # Pad with zeros if needed
                padding = np.zeros((combined.shape[0], self.pca_components - combined.shape[1]))
                combined = np.concatenate([combined, padding], axis=1)
            
            # Normalize
            if combined.std() != 0:
                combined = (combined - combined.mean()) / combined.std()
            
            return combined.flatten()
            
        except Exception as e:
            logging.error(f"Error in feature extraction: {str(e)}")
            return np.zeros(self.pca_components)

    def _select_clients_stratified(self, num_clients: int) -> List[int]:
        """Select clients with stratification across clusters."""
        if not self.client_clusters:
            return list(np.random.choice(
                list(range(self.config['data']['num_clients'])), 
                num_clients, 
                replace=False
            ))
            
        try:
            selected = []
            clusters = set(self.client_clusters.values())
            
            # Select proportionally from each cluster
            for cluster in clusters:
                cluster_clients = [
                    cid for cid, c in self.client_clusters.items() 
                    if c == cluster
                ]
                
                if not cluster_clients:
                    continue
                    
                # Calculate proportional number of clients to select from this cluster
                cluster_count = max(
                    1, 
                    int(num_clients * len(cluster_clients) / 
                        len(self.client_clusters))
                )
                
                # Select clients from this cluster
                cluster_selected = np.random.choice(
                    cluster_clients,
                    min(cluster_count, len(cluster_clients)),
                    replace=False
                ).tolist()
                
                selected.extend(cluster_selected)
            
            # Fill remaining slots if needed
            remaining = num_clients - len(selected)
            if remaining > 0:
                # Get unselected clients
                all_clients = set(range(self.config['data']['num_clients']))
                unselected = list(all_clients - set(selected))
                
                if unselected:
                    additional = np.random.choice(
                        unselected,
                        min(remaining, len(unselected)),
                        replace=False
                    ).tolist()
                    selected.extend(additional)
            
            return selected
            
        except Exception as e:
            logging.error(f"Error in stratified client selection: {str(e)}")
            # Fallback to random selection
            return list(np.random.choice(
                list(range(self.config['data']['num_clients'])), 
                num_clients, 
                replace=False
            ))

    def set_n_clusters(self, n: int):
        """Set number of clusters with validation."""
        if n < self.min_clusters or n > self.max_clusters:
            logging.warning(f"Invalid number of clusters {n}. Using closest valid value.")
            n = max(self.min_clusters, min(n, self.max_clusters))
        self.n_clusters = n

class KMeansCluster(BaseClusteringAlgorithm):
    """Enhanced K-Means clustering implementation."""
    
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.kmeans = None
        self.init = self.config['clustering'].get('kmeans_init', 'k-means++')
        self.max_iter = self.config['clustering'].get('kmeans_max_iter', 300)
        
    def cluster_clients(self, client_features: np.ndarray) -> np.ndarray:
        """Perform K-Means clustering with stability checks."""
        best_inertia = float('inf')
        best_labels = None
        
        # Try multiple initializations
        for _ in range(5):
            kmeans = KMeans(
                n_clusters=self.n_clusters,
                init=self.init,
                max_iter=self.max_iter,
                random_state=None  # Allow different random states
            )
            
            try:
                labels = kmeans.fit_predict(client_features)
                if kmeans.inertia_ < best_inertia:
                    best_inertia = kmeans.inertia_
                    best_labels = labels
                    self.kmeans = kmeans
            except Exception as e:
                logging.warning(f"K-Means clustering attempt failed: {str(e)}")
                continue
        
        if best_labels is None:
            logging.error("All K-Means clustering attempts failed")
            return np.zeros(client_features.shape[0], dtype=int)
            
        return best_labels

class HierarchicalCluster(BaseClusteringAlgorithm):
    """Enhanced Hierarchical clustering implementation."""
    
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.linkages = self.config['clustering'].get('hierarchical_linkages', 
                                                    ['ward', 'complete', 'average'])
        
    def cluster_clients(self, client_features: np.ndarray) -> np.ndarray:
        """Try different linkage methods and select best."""
        best_score = -float('inf')
        best_labels = None
        
        for linkage in self.linkages:
            try:
                clustering = AgglomerativeClustering(
                    n_clusters=self.n_clusters,
                    linkage=linkage
                )
                labels = clustering.fit_predict(client_features)
                score = silhouette_score(client_features, labels)
                
                if score > best_score:
                    best_score = score
                    best_labels = labels
            except Exception as e:
                logging.warning(f"Hierarchical clustering failed with {linkage}: {str(e)}")
                continue
        
        if best_labels is None:
            return np.zeros(client_features.shape[0], dtype=int)
        
        return best_labels

class SpectralCluster(BaseClusteringAlgorithm):
    """Enhanced Spectral clustering implementation."""
    
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.affinities = self.config['clustering'].get('spectral_affinities',
                                                      ['rbf', 'nearest_neighbors'])
        
    def cluster_clients(self, client_features: np.ndarray) -> np.ndarray:
        """Try different affinity methods and select best."""
        best_score = -float('inf')
        best_labels = None
        
        for affinity in self.affinities:
            try:
                clustering = SpectralClustering(
                    n_clusters=self.n_clusters,
                    affinity=affinity,
                    random_state=42
                )
                labels = clustering.fit_predict(client_features)
                score = silhouette_score(client_features, labels)
                
                if score > best_score:
                    best_score = score
                    best_labels = labels
            except Exception as e:
                logging.warning(f"Spectral clustering failed with {affinity}: {str(e)}")
                continue
                
        if best_labels is None:
            return np.zeros(client_features.shape[0], dtype=int)
            
        return best_labels

class GMMCluster(BaseClusteringAlgorithm):
    """Enhanced Gaussian Mixture Model clustering implementation."""
    
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.covariance_types = self.config['clustering'].get('gmm_covariance_types',
                                                            ['full', 'tied', 'diag'])
        self.max_iter = self.config['clustering'].get('gmm_max_iter', 100)
        self.gmm = None
        
    def cluster_clients(self, client_features: np.ndarray) -> np.ndarray:
        """Perform GMM clustering with model selection."""
        best_bic = float('inf')
        best_labels = None
        
        # Try different covariance types if the specified one fails
        for cov_type in self.covariance_types:
            try:
                gmm = GaussianMixture(
                    n_components=self.n_clusters,
                    covariance_type=cov_type,
                    max_iter=self.max_iter,
                    random_state=42
                )
                
                labels = gmm.fit_predict(client_features)
                bic = gmm.bic(client_features)
                
                if bic < best_bic:
                    best_bic = bic
                    best_labels = labels
                    self.gmm = gmm
                    
                if cov_type == self.covariance_type:
                    break  # Stop if the preferred covariance type works
                    
            except Exception as e:
                logging.warning(f"GMM clustering failed with {cov_type} covariance: {str(e)}")
                continue
        
        if best_labels is None:
            logging.error("All GMM clustering attempts failed")
            return np.zeros(client_features.shape[0], dtype=int)
            
        return best_labels

class AdaptiveClusteringManager:
    """Enhanced clustering manager with dynamic adaptation."""
    
    def __init__(self, config_path: str):
        """Initialize clustering manager with advanced features."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        # Basic settings
        self.algorithm_name = self.config['clustering']['algorithm']
        self.algorithm = self._get_clustering_algorithm(config_path)
        self.min_clusters = self.config['clustering'].get('min_clusters', 2)
        self.max_clusters = self.config['clustering'].get('max_clusters', 10)
        
        # Advanced features
        self.drift_threshold = self.config['clustering'].get('drift_threshold', 0.3)
        self.update_frequency = self.config['clustering'].get('update_frequency', 10)
        self.feature_importance = {}
        self.stability_history = []
        self.previous_features = None
        
        # Storage
        self.client_features = {}
        self.client_clusters = None
        self.round_number = 0
        self.last_clustering_round = 0
        self.clustering_performed = False
        
    def should_cluster(self) -> bool:
        """Determine if clustering should be performed."""
        if not self.clustering_performed:
            self.clustering_performed = True
            return True
            
        if self.detect_drift():
            logging.info("Drift detected, triggering re-clustering")
            return True
            
        rounds_since_last = self.round_number - self.last_clustering_round
        if rounds_since_last >= self.update_frequency:
            logging.info("Periodic re-clustering triggered")
            return True
            
        return False
        
    def detect_drift(self) -> bool:
        """Detect distribution drift in client features."""
        if not self.client_features or not self.previous_features:
            return False
            
        current_features = self._get_current_features()
        
        # Compute distribution shift
        drift_score = self._compute_distribution_shift(
            self.previous_features, 
            current_features
        )
        
        self.previous_features = current_features
        return drift_score > self.drift_threshold
        
    def _compute_distribution_shift(self, prev_features: np.ndarray, 
                                  curr_features: np.ndarray) -> float:
        """Compute distribution shift between feature sets."""
        # Normalize features
        prev_norm = StandardScaler().fit_transform(prev_features)
        curr_norm = StandardScaler().fit_transform(curr_features)
        
        # Compute Wasserstein distance for each dimension
        distances = []
        for i in range(prev_norm.shape[1]):
            try:
                dist = wasserstein_distance(prev_norm[:, i], curr_norm[:, i])
                distances.append(dist)
            except Exception as e:
                logging.warning(f"Error computing distance for dimension {i}: {str(e)}")
                continue
        
        return np.mean(distances) if distances else 0.0
    
    def select_clients(self, num_clients: int) -> List[int]:
        """Select clients for training with strategy selection."""
        if not self.client_clusters:
            # Random selection if no clustering is done yet
            return list(np.random.choice(
                list(range(self.config['data']['num_clients'])),
                num_clients,
                replace=False
            ))
        
        return self._select_clients_stratified(num_clients)
    
    def _select_clients_stratified(self, num_clients: int) -> List[int]:
        """Select clients with stratification across clusters."""
        try:
            selected = []
            clusters = set(self.client_clusters.values())
            
            # Select proportionally from each cluster
            for cluster in clusters:
                cluster_clients = [
                    cid for cid, c in self.client_clusters.items() 
                    if c == cluster
                ]
                
                if not cluster_clients:
                    continue
                    
                # Calculate proportional number of clients
                cluster_count = max(
                    1, 
                    int(num_clients * len(cluster_clients) / 
                        len(self.client_clusters))
                )
                
                # Select clients
                cluster_selected = np.random.choice(
                    cluster_clients,
                    min(cluster_count, len(cluster_clients)),
                    replace=False
                ).tolist()
                
                selected.extend(cluster_selected)
            
            # Fill remaining slots if needed
            remaining = num_clients - len(selected)
            if remaining > 0:
                all_clients = set(range(self.config['data']['num_clients']))
                unselected = list(all_clients - set(selected))
                
                if unselected:
                    additional = np.random.choice(
                        unselected,
                        min(remaining, len(unselected)),
                        replace=False
                    ).tolist()
                    selected.extend(additional)
            
            return selected
            
        except Exception as e:
            logging.error(f"Error in stratified client selection: {str(e)}")
            # Fallback to random selection
            return list(np.random.choice(
                list(range(self.config['data']['num_clients'])), 
                num_clients, 
                replace=False
            ))

    def _get_current_features(self) -> np.ndarray:
        """Get current feature matrix for all clients."""
        features_list = [f for f in self.client_features.values() if f is not None]
        if not features_list:
            return np.array([])
        return np.vstack(features_list)

    def estimate_optimal_clusters(self, features: np.ndarray) -> int:
        """Enhanced optimal cluster number estimation with stability metrics."""
        if features.shape[0] < self.min_clusters:
            return self.min_clusters

        max_possible = min(self.max_clusters, features.shape[0] // 2)
        scores = []
        stability_scores = []
        
        # Test each cluster number multiple times
        for k in range(self.min_clusters, max_possible + 1):
            try:
                silhouette_scores = []
                stability = []
                
                # Multiple runs to assess stability
                for _ in range(3):
                    self.algorithm.set_n_clusters(k)
                    labels = self.algorithm.cluster_clients(features)
                    
                    if len(np.unique(labels)) < 2:
                        continue
                        
                    silhouette = silhouette_score(features, labels)
                    silhouette_scores.append(silhouette)
                    
                    # Add noise to test stability
                    noisy_features = features + np.random.normal(0, 0.01, features.shape)
                    noisy_labels = self.algorithm.cluster_clients(noisy_features)
                    stability.append(adjusted_rand_score(labels, noisy_labels))
                
                if silhouette_scores:
                    avg_silhouette = np.mean(silhouette_scores)
                    avg_stability = np.mean(stability)
                    combined_score = 0.7 * avg_silhouette + 0.3 * avg_stability
                    scores.append((k, combined_score))
                    stability_scores.append(avg_stability)
                    
            except Exception as e:
                logging.warning(f"Error computing score for k={k}: {str(e)}")
                continue

        if not scores:
            return self.min_clusters

        # Find best k using combined metrics
        k_values, score_values = zip(*scores)
        best_idx = np.argmax(score_values)
        
        # Check if the solution is stable
        if stability_scores[best_idx] < 0.5:
            logging.warning(f"Selected k={k_values[best_idx]} has low stability")
        
        return k_values[best_idx]

    def update_feature_importance(self) -> None:
        """Update feature importance weights based on clustering performance."""
        if not self.client_clusters:
            return

        current_features = self._get_current_features()
        if current_features.size == 0:
            return

        labels = np.array(list(self.client_clusters.values()))
        
        try:
            # Compute mutual information for each feature
            mi_scores = mutual_info_regression(current_features, labels)
            
            # Normalize scores
            total_score = np.sum(mi_scores)
            if total_score > 0:
                normalized_scores = mi_scores / total_score
                
                # Update feature importance dictionary
                feature_start_idx = 0
                for feat_type in ['model_weights', 'label_distribution', 'data_features']:
                    if feat_type in self.feature_importance:
                        feat_length = len(self.feature_importance[feat_type])
                        feat_scores = normalized_scores[feature_start_idx:feature_start_idx + feat_length]
                        self.feature_importance[feat_type] = np.mean(feat_scores)
                        feature_start_idx += feat_length
        
        except Exception as e:
            logging.error(f"Error updating feature importance: {str(e)}")

    def update_client_features(
        self,
        client_id: int,
        model_weights: Optional[Dict[str, torch.Tensor]] = None,
        data_features: Optional[np.ndarray] = None,
        label_distribution: Optional[np.ndarray] = None
    ) -> None:
        """Update features for a specific client."""
        try:
            # Extract features based on clustering type
            if self.algorithm.clustering_type == 'model_weights' and model_weights is not None:
                features = self.algorithm.extract_features(model_weights=model_weights)
            elif self.algorithm.clustering_type == 'label_distribution' and label_distribution is not None:
                features = self.algorithm.extract_features(label_dist=label_distribution)
            elif self.algorithm.clustering_type == 'data_features' and data_features is not None:
                features = self.algorithm.extract_features(data_features=data_features)
            elif self.algorithm.clustering_type == 'hybrid':
                features = self.algorithm.extract_features(
                    model_weights=model_weights,
                    data_features=data_features,
                    label_dist=label_distribution
                )
            else:
                logging.warning(f"No valid features provided for clustering type: {self.algorithm.clustering_type}")
                return

            # Store features
            self.client_features[client_id] = features

        except Exception as e:
            logging.error(f"Error updating features for client {client_id}: {str(e)}")

    def perform_clustering(self) -> Dict[int, List[int]]:
        """Perform clustering and return client groupings."""
        try:
            if len(self.client_features) == 0:
                raise ValueError("No client features available for clustering")

            # Filter valid features
            valid_features = {cid: f for cid, f in self.client_features.items() 
                            if f is not None and f.size > 0}
            
            if len(valid_features) == 0:
                raise ValueError("No valid client features")

            # Process features
            client_ids = list(valid_features.keys())
            features_array = np.vstack([valid_features[cid] for cid in client_ids])

            # Add small noise to prevent identical features
            features_array += np.random.normal(0, 1e-6, features_array.shape)

            # Estimate optimal number of clusters
            optimal_k = self.estimate_optimal_clusters(features_array)
            self.algorithm.set_n_clusters(optimal_k)

            # Perform clustering
            cluster_assignments = self.algorithm.cluster_clients(features_array)

            # Store clustering results
            self.client_clusters = {
                client_ids[i]: int(cluster_assignments[i])
                for i in range(len(client_ids))
            }

            # Update history and feature importance
            if self.stability_history:
                self.stability_history.append(list(self.client_clusters.values()))
            else:
                self.stability_history = [list(self.client_clusters.values())]
                
            self.update_feature_importance()
            self.last_clustering_round = self.round_number

            # Group clients by cluster
            cluster_groups = {i: [] for i in range(self.algorithm.n_clusters)}
            for cid, cluster_id in self.client_clusters.items():
                cluster_groups[cluster_id].append(cid)

            # Rebalance if needed
            self._rebalance_clusters(cluster_groups)

            logging.info(f"Clustering completed with {self.algorithm.n_clusters} clusters")
            return cluster_groups

        except Exception as e:
            logging.error(f"Clustering failed: {str(e)}")
            # Fallback to single cluster
            return {0: list(range(self.config['data']['num_clients']))}

    def _rebalance_clusters(self, cluster_groups: Dict[int, List[int]]) -> None:
        """Rebalance clusters to prevent empty or oversized clusters."""
        min_size = max(1, len(self.client_features) // (self.max_clusters * 2))
        max_size = len(self.client_features) // self.min_clusters

        # Handle empty or small clusters
        small_clusters = [k for k, v in cluster_groups.items() if len(v) < min_size]
        large_clusters = [k for k, v in cluster_groups.items() if len(v) > max_size]

        for small_cluster in small_clusters:
            if not large_clusters:
                break
                
            large_cluster = large_clusters[0]
            clients_to_move = cluster_groups[large_cluster][:min_size]
            
            # Move clients
            for client in clients_to_move:
                cluster_groups[large_cluster].remove(client)
                cluster_groups[small_cluster].append(client)
                self.client_clusters[client] = small_cluster

            if len(cluster_groups[large_cluster]) <= max_size:
                large_clusters.remove(large_cluster)

    def _get_clustering_algorithm(self, config_path: str) -> BaseClusteringAlgorithm:
        """Get the specified clustering algorithm."""
        algorithms = {
            'kmeans': KMeansCluster,
            'hierarchical': HierarchicalCluster,
            'spectral': SpectralCluster,
            'gmm': GMMCluster
        }
        
        if self.algorithm_name not in algorithms:
            raise ValueError(f"Unsupported clustering algorithm: {self.algorithm_name}")
            
        return algorithms[self.algorithm_name](config_path)

    def evaluate_clustering(self, features: Dict[int, np.ndarray]) -> Dict[str, float]:
        """Evaluate clustering quality with comprehensive metrics."""
        try:
            if not features or not self.client_clusters:
                return self._get_default_metrics()

            feature_matrix = np.vstack(list(features.values()))
            labels = np.array([self.client_clusters[cid] for cid in features.keys()])
            
            # Basic clustering metrics
            metrics = {
                'silhouette_score': float(silhouette_score(feature_matrix, labels)),
                'davies_bouldin_score': float(davies_bouldin_score(feature_matrix, labels)),
                'calinski_harabasz_score': float(calinski_harabasz_score(feature_matrix, labels))
            }

            # Add stability score
            if len(self.stability_history) >= 2:
                metrics['stability_score'] = self._compute_stability_score()

            # Add balance metrics
            balance_metrics = self._compute_balance_metrics()
            metrics.update(balance_metrics)

            return metrics

        except Exception as e:
            logging.error(f"Error computing clustering metrics: {str(e)}")
            return self._get_default_metrics()

    def _get_default_metrics(self) -> Dict[str, float]:
        """Return default metrics when evaluation fails."""
        return {
            'silhouette_score': 0.0,
            'davies_bouldin_score': float('inf'),
            'calinski_harabasz_score': 0.0,
            'stability_score': 0.0,
            'cluster_size_std': 0.0,
            'cluster_size_ratio': 1.0
        }

    def _compute_stability_score(self) -> float:
        """Compute clustering stability score."""
        from sklearn.metrics import adjusted_rand_score
        prev_labels = self.stability_history[-2]
        curr_labels = self.stability_history[-1]
        return float(adjusted_rand_score(prev_labels, curr_labels))

    def _compute_balance_metrics(self) -> Dict[str, float]:
        """Compute metrics for cluster balance."""
        cluster_sizes = []
        for cluster_id in range(self.algorithm.n_clusters):
            size = sum(1 for c in self.client_clusters.values() if c == cluster_id)
            cluster_sizes.append(size)
            
        metrics = {
            'cluster_size_std': float(np.std(cluster_sizes)),
            'cluster_size_ratio': float(min(cluster_sizes) / max(cluster_sizes)) if max(cluster_sizes) > 0 else 0.0
        }
        
        return metrics

    def get_client_cluster(self, client_id: int) -> Optional[int]:
        """Get the cluster assignment for a specific client."""
        return self.client_clusters.get(client_id)
    
    def increment_round(self):
        """Increment the round number."""
        self.round_number += 1

    def save_state(self, path: str):
        """Save clustering state to disk."""
        state = {
            'client_clusters': self.client_clusters,
            'round_number': self.round_number,
            'last_clustering_round': self.last_clustering_round,
            'feature_importance': self.feature_importance,
            'stability_history': self.stability_history,
            'previous_features': self.previous_features
        }
        torch.save(state, path)
        
    def load_state(self, path: str):
        """Load clustering state from disk."""
        if not os.path.exists(path):
            logging.warning(f"No clustering state found at {path}")
            return
            
        state = torch.load(path)
        self.client_clusters = state['client_clusters']
        self.round_number = state['round_number']
        self.last_clustering_round = state['last_clustering_round']
        self.feature_importance = state.get('feature_importance', {})
        self.stability_history = state.get('stability_history', [])
        self.previous_features = state.get('previous_features', None)