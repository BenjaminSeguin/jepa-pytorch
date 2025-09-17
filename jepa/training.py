"""
Training functions and utilities for JEPA.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List


def create_context_target_pairs(sequences, context_len=8, target_len=4):
    """
    Create context-target pairs from sequences for JEPA training
    
    Args:
        sequences: Input sequences (batch_size, seq_len, input_dim)
        context_len: Length of context sequences
        target_len: Length of target sequences
        
    Returns:
        contexts: Context sequences
        targets: Target sequences
    """
    batch_size, seq_len, input_dim = sequences.shape
    
    # Ensure we have enough sequence length
    total_needed = context_len + target_len
    if seq_len < total_needed:
        raise ValueError(f"Sequence length ({seq_len}) must be >= context_len + target_len ({total_needed})")
    
    # Randomly select context start position
    max_start = seq_len - total_needed
    if max_start <= 0:
        start_idx = torch.zeros(batch_size, dtype=torch.long)
    else:
        start_idx = torch.randint(0, max_start + 1, (batch_size,))
    
    contexts = []
    targets = []
    
    for i, start in enumerate(start_idx):
        context = sequences[i, start:start+context_len]
        target = sequences[i, start+context_len:start+context_len+target_len]
        contexts.append(context)
        targets.append(target)
    
    return torch.stack(contexts), torch.stack(targets)


def jepa_loss(predicted_repr, target_repr, temperature=0.1):
    """
    Compute JEPA loss using cosine similarity
    
    Args:
        predicted_repr: Predicted representations
        target_repr: Target representations
        temperature: Temperature parameter for similarity
        
    Returns:
        loss: JEPA loss value
    """
    # Normalize representations
    predicted_repr = F.normalize(predicted_repr, dim=1)
    target_repr = F.normalize(target_repr, dim=1)
    
    # Compute cosine similarity
    similarity = torch.sum(predicted_repr * target_repr, dim=1)
    
    # Convert to loss (negative similarity)
    loss = -similarity.mean()
    
    return loss


def train_jepa(model, dataloader, num_epochs=50, lr=1e-3, context_len=8, target_len=4, momentum=0.99):
    """
    Train the JEPA model
    
    Args:
        model: JEPA model to train
        dataloader: DataLoader for training data
        num_epochs: Number of training epochs
        lr: Learning rate
        context_len: Context sequence length
        target_len: Target sequence length
        momentum: EMA momentum for target encoder
        
    Returns:
        losses: List of training losses
    """
    optimizer = optim.Adam(model.context_encoder.parameters(), lr=lr)
    optimizer.add_param_group({'params': model.predictor.parameters()})
    
    losses = []
    
    model.train()
    for epoch in range(num_epochs):
        epoch_losses = []
        
        for batch_idx, sequences in enumerate(dataloader):
            # Create context-target pairs
            contexts, targets = create_context_target_pairs(
                sequences, context_len=context_len, target_len=target_len
            )
            
            # Forward pass
            context_repr, target_repr, predicted_repr = model(contexts, targets)
            
            # Compute loss
            loss = jepa_loss(predicted_repr, target_repr)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update target encoder with EMA
            model.update_target_encoder(momentum=momentum)
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
    
    return losses


def evaluate_model(model, dataloader, context_len=8, target_len=4):
    """
    Evaluate the trained JEPA model
    
    Args:
        model: Trained JEPA model
        dataloader: DataLoader for evaluation data
        context_len: Context sequence length
        target_len: Target sequence length
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    similarities = []
    losses = []
    
    with torch.no_grad():
        for sequences in dataloader:
            contexts, targets = create_context_target_pairs(
                sequences, context_len=context_len, target_len=target_len
            )
            
            context_repr, target_repr, predicted_repr = model(contexts, targets)
            
            # Compute loss
            loss = jepa_loss(predicted_repr, target_repr)
            losses.append(loss.item())
            
            # Compute similarities
            predicted_repr_norm = F.normalize(predicted_repr, dim=1)
            target_repr_norm = F.normalize(target_repr, dim=1)
            batch_similarities = torch.sum(predicted_repr_norm * target_repr_norm, dim=1)
            similarities.extend(batch_similarities.tolist())
    
    metrics = {
        'avg_loss': np.mean(losses),
        'avg_similarity': np.mean(similarities),
        'std_similarity': np.std(similarities),
        'min_similarity': np.min(similarities),
        'max_similarity': np.max(similarities)
    }
    
    return metrics