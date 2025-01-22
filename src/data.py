# src/data.py

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import numpy as np
from typing import List, Tuple, Dict
import yaml
import logging

class DataManager:
    """Handles dataset loading, preprocessing, and client data distribution."""
    
    def __init__(self, config_path: str):
        """Initialize DataManager with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.dataset_name = self.config['data']['dataset_name']
        self.batch_size = self.config['data']['batch_size']
        self.num_clients = self.config['data']['num_clients']
        self.iid = self.config['data']['iid']
        
        self.train_dataset = None
        self.test_dataset = None
        self.client_data = {}

    def load_dataset(self):
        """Load and preprocess the selected dataset."""
        try:
            if self.dataset_name == "mnist":
                transform = transforms.Compose([transforms.ToTensor()])
                self.train_dataset = datasets.MNIST(
                    root='./data',
                    train=True,
                    download=True,
                    transform=transform
                )
                self.test_dataset = datasets.MNIST(
                    root='./data',
                    train=False,
                    download=True,
                    transform=transform
                )
            elif self.dataset_name == "fashion_mnist":
                transform = transforms.Compose([transforms.ToTensor()])
                self.train_dataset = datasets.FashionMNIST(
                    root='./data',
                    train=True,
                    download=True,
                    transform=transform
                )
                self.test_dataset = datasets.FashionMNIST(
                    root='./data',
                    train=False,
                    download=True,
                    transform=transform
                )
            elif self.dataset_name == "cifar10":
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5),
                                        (0.5, 0.5, 0.5))
                ])
                self.train_dataset = datasets.CIFAR10(
                    root='./data',
                    train=True,
                    download=True,
                    transform=transform
                )
                self.test_dataset = datasets.CIFAR10(
                    root='./data',
                    train=False,
                    download=True,
                    transform=transform
                )
            else:
                raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        except Exception as e:
            logging.error(f"Failed to load dataset: {str(e)}")
            raise RuntimeError("Dataset loading failed")

    def create_client_data(self):
        """Distribute data among clients (IID or non-IID)."""
        if self.iid:
            self._create_iid_clients()
        else:
            self._create_non_iid_clients()
        
        # After creating client_data:
        for cid, subset in self.client_data.items():
            if len(subset) == 0:
                logging.warning(f"Client {cid} has zero samples in non-IID distribution.")

    def _create_iid_clients(self):
        """Create IID data distribution among clients."""
        num_items = len(self.train_dataset)
        indices = torch.randperm(num_items)
        
        # Ensure minimum batch size for each client
        min_items_per_client = max(16, self.batch_size) * 2  # At least 2 batches per client
        client_items = max(num_items // self.num_clients, min_items_per_client)
        
        for i in range(self.num_clients):
            start_idx = i * client_items
            end_idx = min(start_idx + client_items, num_items)
            if end_idx - start_idx < min_items_per_client:
                end_idx = start_idx + min_items_per_client
            self.client_data[i] = Subset(self.train_dataset, indices[start_idx:end_idx])

    def _create_non_iid_clients(self):
        """Create non-IID data distribution among clients (label-based)."""
        import math
        unique_labels = np.unique(self.train_dataset.targets.numpy())
        if len(unique_labels) < 10:
            logging.warning(f"Only {len(unique_labels)} classes present in dataset")
            self.n_clusters = min(self.n_clusters, len(unique_labels))
            
        # Sort data by label
        labels = self.train_dataset.targets.numpy()
        label_indices = {i: np.where(labels == i)[0] for i in range(10)}
        
        # Distribute different proportions of each label to each client
        client_data_indices = [[] for _ in range(self.num_clients)]
        
        for label in range(10):
            # Randomly assign different proportions of this label to each client
            label_size = len(label_indices[label])
            client_proportions = np.random.dirichlet(alpha=[0.5] * self.num_clients)
            client_sizes = (client_proportions * label_size).astype(int)
            
            # Adjust for rounding errors
            client_sizes[-1] = label_size - client_sizes[:-1].sum()
            
            # Distribute indices
            start_idx = 0
            for client_id, size in enumerate(client_sizes):
                client_data_indices[client_id].extend(
                    label_indices[label][start_idx:start_idx + size]
                )
                start_idx += size

        # Create client datasets
        for client_id in range(self.num_clients):
            self.client_data[client_id] = Subset(
                self.train_dataset, 
                client_data_indices[client_id]
            )

    def get_client_loader(self, client_id: int) -> DataLoader:
        """Get DataLoader for a specific client."""
        dataset = self.client_data[client_id]
        actual_batch_size = min(self.batch_size, len(dataset))
        
        return DataLoader(
            dataset,
            batch_size=actual_batch_size,
            shuffle=True,
            drop_last=False  # Include all batches
        )

    def get_test_loader(self) -> DataLoader:
        """Get test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False  # Include all batches
        )