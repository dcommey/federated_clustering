# src/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import yaml

class CNN(nn.Module):
    """Basic CNN architecture suitable for MNIST, Fashion-MNIST, and CIFAR-10."""
    
    def __init__(self, input_channels: int, num_classes: int):
        """
        Initialize the CNN.
        
        Args:
            input_channels (int): Number of input channels (1 for MNIST, 3 for CIFAR-10)
            num_classes (int): Number of output classes
        """
        super(CNN, self).__init__()
        
        # Fix: Add input size validation
        if input_channels not in [1, 3]:
            raise ValueError("Input channels must be 1 or 3")
        if num_classes < 2:
            raise ValueError("Number of classes must be at least 2")
            
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        
        # Add batch normalization for better training stability
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.pool(x)
        
        # Adjust the flattening operation to handle variable batch sizes
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Get model weights as a dictionary."""
        return {name: param.data for name, param in self.named_parameters()}

    def set_weights(self, weights: Dict[str, torch.Tensor]):
        """Set model weights from a dictionary."""
        for name, param in self.named_parameters():
            if name in weights:
                param.data = weights[name]

def create_model(config_path: str) -> CNN:
    """Create a model instance from config."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return CNN(
        input_channels=config['model']['input_channels'],
        num_classes=config['model']['num_classes']
    )