"""
Utility functions for visualization and analysis.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from .training import create_context_target_pairs


def visualize_representations(model, dataloader, num_samples=100, save_path=None):
    """
    Visualize learned representations using PCA projection
    
    Args:
        model: Trained JEPA model
        dataloader: DataLoader for data
        num_samples: Number of samples to visualize
        save_path: Path to save the plot (optional)
    """
    model.eval()
    representations = []
    
    with torch.no_grad():
        for sequences in dataloader:
            contexts, _ = create_context_target_pairs(sequences)
            context_repr = model.context_encoder(contexts)
            representations.append(context_repr)
            
            if len(representations) * contexts.size(0) >= num_samples:
                break
    
    representations = torch.cat(representations, dim=0)[:num_samples]
    
    # Simple 2D projection for visualization (PCA-like)
    U, S, V = torch.pca_lowrank(representations, q=2)
    projected = torch.matmul(representations, V[:, :2])
    
    plt.figure(figsize=(10, 8))
    plt.scatter(projected[:, 0], projected[:, 1], alpha=0.6, s=30)
    plt.title('JEPA Learned Representations (2D PCA Projection)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_curves(losses, save_path=None):
    """
    Plot training loss curves
    
    Args:
        losses: List of training losses
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.title('JEPA Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale often better for loss curves
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_sequences(dataset, num_sequences=6, save_path=None):
    """
    Visualize sample sequences from the dataset
    
    Args:
        dataset: Dataset to visualize
        num_sequences: Number of sequences to show
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(min(num_sequences, len(axes))):
        sequence = dataset[i].numpy()
        ax = axes[i]
        
        # Plot trajectory
        ax.plot(sequence[:, 0], sequence[:, 1], 'b-', alpha=0.7, linewidth=2)
        ax.scatter(sequence[0, 0], sequence[0, 1], c='green', s=100, label='Start', zorder=5)
        ax.scatter(sequence[-1, 0], sequence[-1, 1], c='red', s=100, label='End', zorder=5)
        
        ax.set_title(f'Sequence {i+1}')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def analyze_prediction_quality(model, dataloader, context_len=8, target_len=4, num_examples=5):
    """
    Analyze and visualize prediction quality
    
    Args:
        model: Trained JEPA model
        dataloader: DataLoader for evaluation
        context_len: Context sequence length
        target_len: Target sequence length
        num_examples: Number of examples to analyze
    """
    model.eval()
    
    with torch.no_grad():
        # Get a batch of data
        sequences = next(iter(dataloader))[:num_examples]
        contexts, targets = create_context_target_pairs(
            sequences, context_len=context_len, target_len=target_len
        )
        
        context_repr, target_repr, predicted_repr = model(contexts, targets)
        
        # Compute similarities
        predicted_repr_norm = torch.nn.functional.normalize(predicted_repr, dim=1)
        target_repr_norm = torch.nn.functional.normalize(target_repr, dim=1)
        similarities = torch.sum(predicted_repr_norm * target_repr_norm, dim=1)
        
        # Plot results
        fig, axes = plt.subplots(1, num_examples, figsize=(4*num_examples, 4))
        if num_examples == 1:
            axes = [axes]
        
        for i in range(num_examples):
            ax = axes[i]
            
            # Full sequence
            full_seq = sequences[i].numpy()
            ax.plot(full_seq[:, 0], full_seq[:, 1], 'k-', alpha=0.3, linewidth=1, label='Full sequence')
            
            # Context part
            start_idx = 0  # Simplified for visualization
            context_seq = full_seq[start_idx:start_idx+context_len]
            ax.plot(context_seq[:, 0], context_seq[:, 1], 'b-', linewidth=3, label='Context')
            
            # Target part
            target_seq = full_seq[start_idx+context_len:start_idx+context_len+target_len]
            ax.plot(target_seq[:, 0], target_seq[:, 1], 'r-', linewidth=3, label='Target')
            
            ax.set_title(f'Example {i+1}\nSimilarity: {similarities[i]:.3f}')
            ax.set_xlabel('X coordinate')
            ax.set_ylabel('Y coordinate')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Average similarity: {similarities.mean().item():.4f}")
        print(f"Similarity std: {similarities.std().item():.4f}")


def save_model_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Save model checkpoint
    
    Args:
        model: JEPA model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)


def load_model_checkpoint(model, optimizer, filepath):
    """
    Load model checkpoint
    
    Args:
        model: JEPA model
        optimizer: Optimizer
        filepath: Path to checkpoint file
        
    Returns:
        epoch: Epoch number
        loss: Loss value
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return epoch, loss