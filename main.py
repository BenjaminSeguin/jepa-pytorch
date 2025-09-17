import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class ToySequenceDataset(Dataset):
    """
    Toy dataset generating sequences of 2D points following simple patterns
    """
    def __init__(self, num_samples=1000, seq_len=20, pattern_type='circle'):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.pattern_type = pattern_type
        self.data = self._generate_data()
    
    def _generate_data(self):
        sequences = []
        for _ in range(self.num_samples):
            if self.pattern_type == 'circle':
                # Generate circular motion with some noise
                t = np.linspace(0, 2*np.pi, self.seq_len)
                radius = np.random.uniform(0.5, 2.0)
                center_x = np.random.uniform(-1, 1)
                center_y = np.random.uniform(-1, 1)
                x = center_x + radius * np.cos(t) + np.random.normal(0, 0.1, self.seq_len)
                y = center_y + radius * np.sin(t) + np.random.normal(0, 0.1, self.seq_len)
                sequence = np.stack([x, y], axis=1)
            
            elif self.pattern_type == 'linear':
                # Generate linear motion with some noise
                start_x = np.random.uniform(-2, 2)
                start_y = np.random.uniform(-2, 2)
                vel_x = np.random.uniform(-1, 1)
                vel_y = np.random.uniform(-1, 1)
                t = np.arange(self.seq_len)
                x = start_x + vel_x * t + np.random.normal(0, 0.1, self.seq_len)
                y = start_y + vel_y * t + np.random.normal(0, 0.1, self.seq_len)
                sequence = np.stack([x, y], axis=1)
            
            sequences.append(sequence.astype(np.float32))
        return sequences
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])

class Encoder(nn.Module):
    """
    Context Encoder - encodes input sequences into representations
    """
    def __init__(self, input_dim=2, hidden_dim=64, repr_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.repr_dim = repr_dim
        
        # LSTM encoder
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2)
        
        # Projection to representation space
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, repr_dim)
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Take the last output for each sequence
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        representation = self.projection(last_output)
        return representation

class Predictor(nn.Module):
    """
    Predictor network - predicts target representations from context representations
    """
    def __init__(self, repr_dim=32, hidden_dim=64):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, repr_dim)
        )
    
    def forward(self, context_repr):
        return self.predictor(context_repr)

class JEPA(nn.Module):
    """
    Joint-Embedding Predictive Architecture
    """
    def __init__(self, input_dim=2, hidden_dim=64, repr_dim=32):
        super().__init__()
        self.context_encoder = Encoder(input_dim, hidden_dim, repr_dim)
        self.target_encoder = Encoder(input_dim, hidden_dim, repr_dim)
        self.predictor = Predictor(repr_dim, hidden_dim)
        
        # Initialize target encoder with same weights as context encoder
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        
        # Target encoder parameters will be updated with EMA
        for param in self.target_encoder.parameters():
            param.requires_grad = False
    
    def forward(self, context, target):
        # Encode context and target
        context_repr = self.context_encoder(context)
        target_repr = self.target_encoder(target)
        
        # Predict target representation from context
        predicted_repr = self.predictor(context_repr)
        
        return context_repr, target_repr, predicted_repr
    
    def update_target_encoder(self, momentum=0.99):
        """
        Update target encoder parameters with exponential moving average
        """
        with torch.no_grad():
            for target_param, context_param in zip(
                self.target_encoder.parameters(), 
                self.context_encoder.parameters()
            ):
                target_param.data = momentum * target_param.data + (1 - momentum) * context_param.data

def create_context_target_pairs(sequences, context_len=8, target_len=4):
    """
    Create context-target pairs from sequences for JEPA training
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
    """
    # Normalize representations
    predicted_repr = F.normalize(predicted_repr, dim=1)
    target_repr = F.normalize(target_repr, dim=1)
    
    # Compute cosine similarity
    similarity = torch.sum(predicted_repr * target_repr, dim=1)
    
    # Convert to loss (negative similarity)
    loss = -similarity.mean()
    
    return loss

def train_jepa(model, dataloader, num_epochs=50, lr=1e-3):
    """
    Train the JEPA model
    """
    optimizer = optim.Adam(model.context_encoder.parameters(), lr=lr)
    optimizer.add_param_group({'params': model.predictor.parameters()})
    
    losses = []
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        for batch_idx, sequences in enumerate(dataloader):
            # Create context-target pairs
            contexts, targets = create_context_target_pairs(sequences)
            
            # Forward pass
            context_repr, target_repr, predicted_repr = model(contexts, targets)
            
            # Compute loss
            loss = jepa_loss(predicted_repr, target_repr)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update target encoder with EMA
            model.update_target_encoder()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
    
    return losses

def visualize_representations(model, dataloader, num_samples=100):
    """
    Visualize learned representations using t-SNE
    """
    model.eval()
    representations = []
    patterns = []
    
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
    plt.scatter(projected[:, 0], projected[:, 1], alpha=0.6)
    plt.title('JEPA Learned Representations (2D Projection)')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    # Create datasets
    print("Creating datasets...")
    train_dataset = ToySequenceDataset(num_samples=800, seq_len=20, pattern_type='circle')
    val_dataset = ToySequenceDataset(num_samples=200, seq_len=20, pattern_type='circle')
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create JEPA model
    print("Initializing JEPA model...")
    model = JEPA(input_dim=2, hidden_dim=64, repr_dim=32)
    
    # Train the model
    print("Training JEPA...")
    losses = train_jepa(model, train_loader, num_epochs=50)
    
    # Plot training losses
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('JEPA Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Visualize learned representations
    print("Visualizing learned representations...")
    visualize_representations(model, val_loader)
    
    # Test prediction capability
    print("\nTesting prediction capability...")
    model.eval()
    with torch.no_grad():
        sample_sequences = next(iter(val_loader))[:4]  # Take 4 samples
        contexts, targets = create_context_target_pairs(sample_sequences)
        
        context_repr, target_repr, predicted_repr = model(contexts, targets)
        
        # Compute similarities
        predicted_repr_norm = F.normalize(predicted_repr, dim=1)
        target_repr_norm = F.normalize(target_repr, dim=1)
        similarities = torch.sum(predicted_repr_norm * target_repr_norm, dim=1)
        
        print(f"Prediction similarities: {similarities}")
        print(f"Mean similarity: {similarities.mean().item():.4f}")
    
    return model, losses

if __name__ == "__main__":
    # Run the example
    model, losses = main()
    
    print("\nJEPA training completed!")
    print("The model has learned to predict future sequence representations")
    print("from context representations in a self-supervised manner.")