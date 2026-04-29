"""
ECE 565 - Computer Vision Final Project (Stage 1)
Training Execution Script

Handles the training loop, validation, logging, and checkpointing for the 
classification backbones on CIFAR-100.
"""

import argparse
import os
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import (get_network, get_training_dataloader, get_test_dataloader, 
                   WarmUpLR, most_recent_folder, most_recent_weights, 
                   last_epoch, best_acc_weights)


def train_epoch(epoch: int, net: nn.Module, loader: DataLoader, criterion: nn.Module, 
                optimizer: optim.Optimizer, writer: SummaryWriter, device: torch.device, 
                warmup_scheduler: WarmUpLR, args: argparse.Namespace) -> None:
    """Executes a single epoch of training."""
    start_time = time.time()
    net.train()
    
    for batch_index, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Update metrics
        n_iter = (epoch - 1) * len(loader) + batch_index + 1
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    finish_time = time.time()
    print(f'Epoch {epoch} training time consumed: {finish_time - start_time:.2f}s')


@torch.no_grad()
def eval_epoch(epoch: int, net: nn.Module, loader: DataLoader, criterion: nn.Module, 
               writer: SummaryWriter, device: torch.device, tb: bool = True) -> float:
    """Evaluates the model on the test dataset."""
    start_time = time.time()
    net.eval()

    test_loss = 0.0
    correct = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = net(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()

    dataset_size = len(loader.dataset)
    avg_loss = test_loss / dataset_size
    accuracy = correct / dataset_size
    finish_time = time.time()

    print(f'Test set: Epoch: {epoch}, Average loss: {avg_loss:.4f}, '
          f'Accuracy: {accuracy:.4f}, Time consumed: {finish_time - start_time:.2f}s\n')

    if tb:
        writer.add_scalar('Test/Average loss', avg_loss, epoch)
        writer.add_scalar('Test/Accuracy', accuracy, epoch)

    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train network on CIFAR-100")
    parser.add_argument('-net', type=str, default='mobilenet', help='Network architecture type')
    parser.add_argument('-gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('-b', type=int, default=128, help='Batch size for dataloader')
    parser.add_argument('-warm', type=int, default=5, help='Warm up training phase (epochs)')
    parser.add_argument('-lr', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('-resume', action='store_true', help='Resume training from recent checkpoint')
    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")

    net = get_network(args).to(device)

    # Data Loading
    train_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD,
        num_workers=4, batch_size=args.b, shuffle=True
    )
    test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD,
        num_workers=4, batch_size=args.b, shuffle=False
    )

    # Optimization Configuration
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2)
    warmup_scheduler = WarmUpLR(optimizer, len(train_loader) * args.warm)

    # Logging and Checkpoint Setup
    checkpoint_dir = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    resume_epoch = 0
    best_acc = 0.0

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('No recent folder found to resume from.')
        
        checkpoint_dir = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)
        recent_weights_file = most_recent_weights(checkpoint_dir)
        weights_path = os.path.join(checkpoint_dir, recent_weights_file)
        
        print(f'[*] Loading weights file {weights_path} to resume training...')
        net.load_state_dict(torch.load(weights_path, map_location=device))
        resume_epoch = last_epoch(checkpoint_dir)

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_template = os.path.join(checkpoint_dir, '{net}-{epoch}-{type}.pth')

    os.makedirs(settings.LOG_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(settings.LOG_DIR, args.net, settings.TIME_NOW))

    # Training Loop
    for epoch in range(1, settings.EPOCH + 1):
        if args.resume and epoch <= resume_epoch:
            continue

        train_epoch(epoch, net, train_loader, criterion, optimizer, writer, device, warmup_scheduler, args)
        
        # Step the main scheduler only after warmup ends
        if epoch > args.warm:
            train_scheduler.step()

        acc = eval_epoch(epoch, net, test_loader, criterion, writer, device)

        # Save Best Model
        if epoch > settings.MILESTONES[1] and acc > best_acc:
            weights_path = checkpoint_template.format(net=args.net, epoch=f"{epoch:03d}", type='best')
            print(f'[*] Saving best weights to {weights_path}')
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        # Save Regular Checkpoints
        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_template.format(net=args.net, epoch=f"{epoch:03d}", type='regular')
            print(f'[*] Saving regular weights to {weights_path}')
            torch.save(net.state_dict(), weights_path)

    writer.close()


if __name__ == '__main__':
    main()