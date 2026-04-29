"""
ECE 565 - Computer Vision Final Project (Stage 1)
Model Evaluation Script

Calculates Top-1 and Top-5 accuracy/error rates for a saved model 
on the CIFAR-100 test dataset.
"""

import argparse
import torch
from tqdm import tqdm

from conf import settings
from utils import get_network, get_test_dataloader


def evaluate_model(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"[*] Evaluating on device: {device}")

    # Initialize model and load weights
    net = get_network(args)
    net.load_state_dict(torch.load(args.weights, map_location=device))
    net.to(device)
    net.eval()

    test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD,
        num_workers=4, batch_size=args.b, shuffle=False
    )

    correct_1 = 0.0
    correct_5 = 0.0
    dataset_size = len(test_loader.dataset)

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Processing Test Set"):
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, pred = outputs.topk(5, 1, largest=True, sorted=True)

            # Reshape labels to match prediction dimensions for comparison
            labels_expanded = labels.view(labels.size(0), -1).expand_as(pred)
            correct = pred.eq(labels_expanded).float()

            # Compute Top-1
            correct_1 += correct[:, :1].sum().item()
            
            # Compute Top-5
            correct_5 += correct[:, :5].sum().item()

    # Calculate metrics
    top_1_acc = (correct_1 / dataset_size) * 100
    top_1_err = 100 - top_1_acc
    top_5_acc = (correct_5 / dataset_size) * 100
    top_5_err = 100 - top_5_acc
    total_params = sum(p.numel() for p in net.parameters())

    print("\n" + "="*30)
    print("      EVALUATION RESULTS")
    print("="*30)
    print(f"Model Parameters: {total_params:,}")
    print(f"Top 1 Accuracy:   {top_1_acc:.2f}%")
    print(f"Top 1 Error:      {top_1_err:.2f}%")
    print(f"Top 5 Accuracy:   {top_5_acc:.2f}%")
    print(f"Top 5 Error:      {top_5_err:.2f}%")
    print("="*30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate network on CIFAR-100 test set")
    parser.add_argument('-net', type=str, default='mobilenet', help='Network architecture type')
    parser.add_argument('-weights', type=str, required=True, help='Path to the .pth weights file')
    parser.add_argument('-gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('-b', type=int, default=64, help='Batch size for dataloader')
    
    args = parser.parse_args()
    evaluate_model(args)