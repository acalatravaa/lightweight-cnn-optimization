"""
ECE 565 - Computer Vision Final Project (Stage 1)
CIFAR-100 Dataset Loaders

Custom PyTorch Dataset implementations for loading CIFAR-100 binary files.
"""

import os
import pickle
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset


class _CIFAR100Base(Dataset):
    """
    Base class for CIFAR-100 dataset loading to prevent code duplication.
    Handles the binary unpacking and efficient array reshaping.
    """
    def __init__(self, file_path: str, transform: Optional[Callable] = None):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CIFAR-100 data file not found at: {file_path}")

        with open(file_path, 'rb') as f:
            self.data_dict: Dict[bytes, Any] = pickle.load(f, encoding='bytes')
            
        self.transform = transform
        self.labels = self.data_dict[b'fine_labels']
        self.images = self.data_dict[b'data']

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[int, Any]:
        label = self.labels[index]
        
        # Vectorized reshape: converts flat 3072-element array to (3, 32, 32)
        # then transposes to (32, 32, 3) which is the standard image format.
        image = self.images[index].reshape(3, 32, 32).transpose(1, 2, 0)

        if self.transform:
            image = self.transform(image)
            
        return label, image


class CIFAR100Train(_CIFAR100Base):
    """CIFAR-100 training dataset."""
    
    def __init__(self, path: str, transform: Optional[Callable] = None):
        super().__init__(os.path.join(path, 'train'), transform)


class CIFAR100Test(_CIFAR100Base):
    """CIFAR-100 test dataset."""
    
    def __init__(self, path: str, transform: Optional[Callable] = None):
        super().__init__(os.path.join(path, 'test'), transform)