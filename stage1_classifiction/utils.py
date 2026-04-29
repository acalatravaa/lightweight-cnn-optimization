"""
Provides data loaders, model builders, learning rate schedulers, 
and checkpoint management functions.
"""

import os
import re
import argparse
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from conf import settings
from dataset import CIFAR100Train, CIFAR100Test


def get_network(args: argparse.Namespace) -> nn.Module:
    """
    Instantiates and returns the specified network architecture.

    Args:
        args (argparse.Namespace): Command line arguments containing the 'net' attribute.

    Returns:
        nn.Module: The instantiated PyTorch model.
    """
    if args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
        
    elif args.net == 'base_linear':
        from models.mobilenet import build_modified_baseline
        net = build_modified_baseline(use_linear=True, use_eca=False, use_shuffle=False)

    elif args.net == 'base_eca':
        from models.mobilenet import build_modified_baseline
        net = build_modified_baseline(use_linear=False, use_eca=True, use_shuffle=False)

    elif args.net == 'base_shuffle':
        from models.mobilenet import build_modified_baseline
        net = build_modified_baseline(use_linear=False, use_eca=False, use_shuffle=True)
        
    elif args.net == 'wide_linear':
        from models.mobilenet import WideLinearMobileNet
        net = WideLinearMobileNet()
    else:
        raise ValueError(f"Architecture '{args.net}' is not recognized.")

    return net


def get_training_dataloader(mean: Tuple[float, ...], std: Tuple[float, ...], batch_size: int = 16, 
                            num_workers: int = 2, shuffle: bool = True) -> DataLoader:
    """Returns a DataLoader for the CIFAR-100 training set with AutoAugment."""
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    cifar100_training = CIFAR100Train(settings.DATA_DIR, transform=transform_train)
    
    return DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size
    )


def get_test_dataloader(mean: Tuple[float, ...], std: Tuple[float, ...], batch_size: int = 16, 
                        num_workers: int = 2, shuffle: bool = True) -> DataLoader:
    """Returns a DataLoader for the CIFAR-100 test set."""
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    cifar100_test = CIFAR100Test(settings.DATA_DIR, transform=transform_test)
    
    return DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size
    )


class WarmUpLR(_LRScheduler):
    """
    Linearly warms up the learning rate during early epochs.
    Prevents early gradient spikes before the standard step-decay schedule takes over.
    """
    def __init__(self, optimizer: torch.optim.Optimizer, total_iters: int, last_epoch: int = -1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        return [
            base_lr * self.last_epoch / (self.total_iters + 1e-8) 
            for base_lr in self.base_lrs
        ]


# =============================================================================
# --- CHECKPOINT MANAGEMENT UTILITIES ---
# =============================================================================

def most_recent_folder(net_weights_dir: str, fmt: str) -> Optional[str]:
    """Retrieves the most recently created timestamped directory for a model."""
    if not os.path.exists(net_weights_dir):
        return None
        
    folders = os.listdir(net_weights_dir)
    if not folders:
        return None

    # Filter out non-directory files and sort by modification time
    folders = [f for f in folders if os.path.isdir(os.path.join(net_weights_dir, f))]
    folders.sort(key=lambda f: os.path.getmtime(os.path.join(net_weights_dir, f)))
    
    return folders[-1] if folders else None


def most_recent_weights(weights_folder: str) -> Optional[str]:
    """
    Retrieves the filename of the most recent regular epoch checkpoint.
    Expects format: '{epoch}-regular.pth'
    """
    if not os.path.exists(weights_folder):
        return None
        
    weight_files = os.listdir(weights_folder)
    if not weight_files:
        return None

    # Filter for standard epoch checkpoints
    regex_str = r'(\d+)-regular\.pth'
    valid_files = [f for f in weight_files if re.search(regex_str, f)]
    
    if not valid_files:
        return None

    def get_epoch(filename: str) -> int:
        match = re.search(regex_str, filename)
        return int(match.group(1)) if match else -1

    valid_files.sort(key=get_epoch)
    return valid_files[-1]


def last_epoch(weights_folder: str) -> int:
    """Extracts the epoch integer from the most recent weight file."""
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
        raise FileNotFoundError(f"No recent weights found in {weights_folder}")
        
    match = re.search(r'(\d+)-regular\.pth', weight_file)
    if not match:
        raise ValueError(f"Filename '{weight_file}' does not match expected epoch format.")
        
    return int(match.group(1))


def best_acc_weights(weights_folder: str) -> Optional[str]:
    """Retrieves the filename of the checkpoint with the highest validation accuracy."""
    if not os.path.exists(weights_folder):
        return None
        
    files = os.listdir(weights_folder)
    if not files:
        return None

    # Filter for best epoch checkpoints
    regex_str = r'(\d+)-best\.pth'
    best_files = [f for f in files if re.search(regex_str, f)]
    
    if not best_files:
        return None

    def get_epoch(filename: str) -> int:
        match = re.search(regex_str, filename)
        return int(match.group(1)) if match else -1

    best_files.sort(key=get_epoch)
    return best_files[-1]