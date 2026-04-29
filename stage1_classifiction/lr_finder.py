"""
ECE 565 - Computer Vision Final Project (Stage 1)
Learning Rate Finder

Implements an exponentially increasing learning rate schedule to empirically 
determine the optimal initial learning rate before full model training.
"""

import argparse
import os
from typing import List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

# Assuming these are accessible in your PYTHONPATH based on your structure
from conf import settings
from utils import get_training_dataloader, get_network


class ExponentialLRScheduler(_LRScheduler):
    """
    Exponentially increases the learning rate from a base value to a maximum value
    over a specified number of iterations. Used to plot the loss landscape.

    Args:
        optimizer (optim.Optimizer): Wrapped optimizer.
        max_lr (float): The peak learning rate to reach.
        num_iter (int): Total number of iterations (batches) to reach max_lr.
        last_epoch (int): The index of the last epoch. Default: -1.
    """
    def __init__(self, optimizer: optim.Optimizer, max_lr: float = 10.0, num_iter: int = 100, last_epoch: int = -1):
        self.total_iters = num_iter
        self.max_lr = max_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Calculates the exponentially increasing learning rate for each parameter group."""
        return [
            base_lr * (self.max_lr / base_lr) ** (self.last_epoch / (self.total_iters + 1e-32))
            for base_lr in self.base_lrs
        ]


def plot_lr_landscape(learning_rates: List[float], losses: List[float], save_path: str = 'lr_finder_result.jpg') -> None:
    """
    Plots the learning rate against the recorded losses and saves the figure.
    Trims the first and last few outliers for a cleaner visualization.
    """
    # Slice off early noise and late explosions for better graph scaling
    trim_start, trim_end = 10, -5
    if len(learning_rates) > (trim_start + abs(trim_end)):
        learning_rates = learning_rates[trim_start:trim_end]
        losses = losses[trim_start:trim_end]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(learning_rates, losses, linewidth=2)
    ax.set_xlabel('Learning Rate (Log Scale)', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Learning Rate Finder Range', fontsize=14)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
    ax.grid(True, which="both", ls="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    print(f"[*] Learning rate landscape plot saved to: {save_path}")
    plt.close(fig)


def find_learning_rate(args: argparse.Namespace) -> None:
    """Core logic to iterate through batches and record loss at exponentially increasing LRs."""
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"[*] Executing LR Finder on device: {device}")

    # Setup Dataloader
    train_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
    )

    # Setup Model, Loss, and Optimizer
    net = get_network(args).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    
    # Setup LR Scheduler
    lr_scheduler = ExponentialLRScheduler(optimizer, max_lr=args.max_lr, num_iter=args.num_iter)
    
    # Calculate required epochs to fulfill iteration count
    epochs = int(args.num_iter / len(train_loader)) + 1

    iteration = 0
    learning_rates: List[float] = []
    losses: List[float] = []

    net.train()
    
    print(f"[*] Starting LR finder for {args.num_iter} iterations...")
    
    for epoch in range(epochs):
        for batch_index, (images, labels) in enumerate(train_loader):
            if iteration > args.num_iter:
                break

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)

            # Abort if loss explodes
            if torch.isnan(loss).any():
                print("[!] Loss exploded to NaN. Stopping search early.")
                iteration += int(1e8) # Force outer loop break
                break

            loss.backward()
            optimizer.step()
            
            # Step scheduler AFTER optimizer.step() to comply with modern PyTorch standards
            lr_scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            
            # Record metrics
            learning_rates.append(current_lr)
            losses.append(loss.item())
            
            if iteration % 10 == 0 or iteration == args.num_iter:
                samples_trained = batch_index * args.b + len(images)
                print(f"Iteration: {iteration:03d} [{samples_trained}/{len(train_loader.dataset)}]\t"
                      f"Loss: {loss.item():.4f}\tLR: {current_lr:.8f}")

            iteration += 1

        if iteration > args.num_iter:
            break

    plot_lr_landscape(learning_rates, losses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch Learning Rate Finder")
    parser.add_argument('-net', type=str, required=True, help='Network architecture type')
    parser.add_argument('-b', type=int, default=64, help='Batch size for dataloader')
    parser.add_argument('-base_lr', type=float, default=1e-7, help='Minimum starting learning rate')
    parser.add_argument('-max_lr', type=float, default=10.0, help='Maximum learning rate')
    parser.add_argument('-num_iter', type=int, default=100, help='Number of iterations to search over')
    parser.add_argument('-gpu', action='store_true', help='Use GPU if available')
    
    args = parser.parse_args()
    
    find_learning_rate(args)