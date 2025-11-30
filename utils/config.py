import torch
from fully_connected_basics.datasets import get_mnist_loaders, get_cifar_loaders
import os

CONFIGS = {
    'mnist': {
        'input_size': 28 * 28,
        'num_classes': 10
    },
    'cifar': {
        'input_size': 32 * 32 * 3,
        'num_classes': 10
    }
}

DEVICE = str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

LOADERS = {
    'mnist': get_mnist_loaders(),
    'cifar': get_cifar_loaders()
}

PROJECT_ROOT = os.getcwd()