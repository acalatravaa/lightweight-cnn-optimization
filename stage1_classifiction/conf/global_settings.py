"""
ECE 565 - Computer Vision Final Project
Global Configurations and Default Hyperparameters
"""

import os
from datetime import datetime
from typing import Tuple, List

# Dataset Normalization Statistics (CIFAR-100)
CIFAR100_TRAIN_MEAN: Tuple[float, float, float] = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD: Tuple[float, float, float] = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

# Directory Paths
# Use environment variables if set (useful for Docker/cloud training), otherwise default to local paths
CHECKPOINT_PATH: str = os.getenv('CHECKPOINT_PATH', 'checkpoint')
LOG_DIR: str = os.getenv('LOG_DIR', 'runs')
DATA_DIR: str = os.getenv('DATA_DIR', 'data')

# Training Hyperparameters
EPOCH: int = 100
MILESTONES: List[int] = [50, 75, 90]
SAVE_EPOCH: int = 25

# Formatting
DATE_FORMAT: str = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
TIME_NOW: str = datetime.now().strftime(DATE_FORMAT)