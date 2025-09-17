"""
JEPA: Joint-Embedding Predictive Architecture

A PyTorch implementation of Meta's Joint-Embedding Predictive Architecture 
for self-supervised representation learning.
"""

from .models import JEPA, Encoder, Predictor
from .datasets import ToySequenceDataset
from .training import train_jepa, jepa_loss, create_context_target_pairs
from .utils import visualize_representations

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "JEPA",
    "Encoder", 
    "Predictor",
    "ToySequenceDataset",
    "train_jepa",
    "jepa_loss",
    "create_context_target_pairs",
    "visualize_representations"
]