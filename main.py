"""
Main training script for JEPA (Joint-Embedding Predictive Architecture)

This script demonstrates the complete training pipeline for JEPA on toy sequential data.
"""

import torch
import numpy as np
import random
from torch.utils.data import DataLoader

from jepa.models import JEPA
from jepa.datasets import ToySequenceDataset
from jepa.training import train_jepa, evaluate_model
from jepa.utils import (
    visualize_representations, 
    plot_training_curves, 
    visualize_sequences,
    analyze_prediction_quality
)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def main():
    """Main training function"""
    print("=" * 60)
    print("JEPA: Joint-Embedding Predictive Architecture")
    print("=" * 60)
    
    # Configuration
    config = {
        'num_train_samples': 800,
        'num_val_samples': 200,
        'seq_len': 20,
        'pattern_type': 'circle',
        'batch_size': 32,
        'input_dim': 2,
        'hidden_dim': 64,
        'repr_dim': 32,
        'context_len': 8,
        'target_len': 4,
        'num_epochs': 50,
        'learning_rate': 1e-3,
        'momentum': 0.99
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = ToySequenceDataset(
        num_samples=config['num_train_samples'], 
        seq_len=config['seq_len'], 
        pattern_type=config['pattern_type']
    )
    val_dataset = ToySequenceDataset(
        num_samples=config['num_val_samples'], 
        seq_len=config['seq_len'], 
        pattern_type=config['pattern_type']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    print(f"Train dataset: {len(train_dataset)} sequences")
    print(f"Validation dataset: {len(val_dataset)} sequences")
    print()
    
    # Visualize sample sequences
    print("Visualizing sample sequences...")
    visualize_sequences(train_dataset, num_sequences=6)
    
    # Create JEPA model
    print("Initializing JEPA model...")
    model = JEPA(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        repr_dim=config['repr_dim']
    )
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {total_params:,} trainable parameters")
    print()
    
    # Train the model
    print("Training JEPA...")
    losses = train_jepa(
        model=model,
        dataloader=train_loader,
        num_epochs=config['num_epochs'],
        lr=config['learning_rate'],
        context_len=config['context_len'],
        target_len=config['target_len'],
        momentum=config['momentum']
    )
    
    print(f"Training completed! Final loss: {losses[-1]:.4f}")
    print()
    
    # Plot training curves
    print("Plotting training curves...")
    plot_training_curves(losses)
    
    # Evaluate the model
    print("Evaluating model on validation set...")
    metrics = evaluate_model(
        model=model,
        dataloader=val_loader,
        context_len=config['context_len'],
        target_len=config['target_len']
    )
    
    print("Validation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    print()
    
    # Visualize learned representations
    print("Visualizing learned representations...")
    visualize_representations(model, val_loader, num_samples=200)
    
    # Analyze prediction quality
    print("Analyzing prediction quality...")
    analyze_prediction_quality(
        model=model,
        dataloader=val_loader,
        context_len=config['context_len'],
        target_len=config['target_len'],
        num_examples=5
    )
    
    print("=" * 60)
    print("JEPA training and evaluation completed successfully!")
    print("The model has learned to predict future sequence representations")
    print("from context representations in a self-supervised manner.")
    print("=" * 60)
    
    return model, losses, metrics


if __name__ == "__main__":
    # Run the complete pipeline
    model, losses, metrics = main()