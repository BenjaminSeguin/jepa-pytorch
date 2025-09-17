"""
Dataset classes for JEPA training.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import random


class ToySequenceDataset(Dataset):
    """
    Toy dataset generating sequences of 2D points following simple patterns
    
    Args:
        num_samples (int): Number of sequences to generate
        seq_len (int): Length of each sequence
        pattern_type (str): Type of pattern ('circle', 'linear')
    """
    def __init__(self, num_samples=1000, seq_len=20, pattern_type='circle'):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.pattern_type = pattern_type
        self.data = self._generate_data()
    
    def _generate_data(self):
        """Generate sequences based on pattern type"""
        sequences = []
        for _ in range(self.num_samples):
            if self.pattern_type == 'circle':
                sequence = self._generate_circular_motion()
            elif self.pattern_type == 'linear':
                sequence = self._generate_linear_motion()
            else:
                raise ValueError(f"Unknown pattern type: {self.pattern_type}")
            
            sequences.append(sequence.astype(np.float32))
        return sequences
    
    def _generate_circular_motion(self):
        """Generate circular motion with noise"""
        t = np.linspace(0, 2*np.pi, self.seq_len)
        radius = np.random.uniform(0.5, 2.0)
        center_x = np.random.uniform(-1, 1)
        center_y = np.random.uniform(-1, 1)
        x = center_x + radius * np.cos(t) + np.random.normal(0, 0.1, self.seq_len)
        y = center_y + radius * np.sin(t) + np.random.normal(0, 0.1, self.seq_len)
        return np.stack([x, y], axis=1)
    
    def _generate_linear_motion(self):
        """Generate linear motion with noise"""
        start_x = np.random.uniform(-2, 2)
        start_y = np.random.uniform(-2, 2)
        vel_x = np.random.uniform(-1, 1)
        vel_y = np.random.uniform(-1, 1)
        t = np.arange(self.seq_len)
        x = start_x + vel_x * t + np.random.normal(0, 0.1, self.seq_len)
        y = start_y + vel_y * t + np.random.normal(0, 0.1, self.seq_len)
        return np.stack([x, y], axis=1)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])


def create_mixed_dataset(num_samples=1000, seq_len=20, split_ratio=0.5):
    """
    Create a mixed dataset with both circular and linear patterns
    
    Args:
        num_samples (int): Total number of samples
        seq_len (int): Sequence length
        split_ratio (float): Ratio of circular to linear patterns
        
    Returns:
        Dataset with mixed patterns
    """
    circular_samples = int(num_samples * split_ratio)
    linear_samples = num_samples - circular_samples
    
    circular_data = ToySequenceDataset(circular_samples, seq_len, 'circle')
    linear_data = ToySequenceDataset(linear_samples, seq_len, 'linear')
    
    # Combine datasets
    combined_data = circular_data.data + linear_data.data
    random.shuffle(combined_data)
    
    # Create new dataset object
    mixed_dataset = ToySequenceDataset.__new__(ToySequenceDataset)
    mixed_dataset.num_samples = num_samples
    mixed_dataset.seq_len = seq_len
    mixed_dataset.pattern_type = 'mixed'
    mixed_dataset.data = combined_data
    
    return mixed_dataset