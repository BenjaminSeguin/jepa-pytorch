"""
JEPA model architectures and components.
"""

import torch
import torch.nn as nn


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
    
    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden dimension for encoders
        repr_dim (int): Representation dimension
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
        """
        Forward pass of JEPA model
        
        Args:
            context: Context sequences (batch_size, context_len, input_dim)
            target: Target sequences (batch_size, target_len, input_dim)
            
        Returns:
            context_repr: Context representations
            target_repr: Target representations  
            predicted_repr: Predicted target representations
        """
        # Encode context and target
        context_repr = self.context_encoder(context)
        target_repr = self.target_encoder(target)
        
        # Predict target representation from context
        predicted_repr = self.predictor(context_repr)
        
        return context_repr, target_repr, predicted_repr
    
    def update_target_encoder(self, momentum=0.99):
        """
        Update target encoder parameters with exponential moving average
        
        Args:
            momentum (float): EMA momentum parameter
        """
        with torch.no_grad():
            for target_param, context_param in zip(
                self.target_encoder.parameters(), 
                self.context_encoder.parameters()
            ):
                target_param.data = momentum * target_param.data + (1 - momentum) * context_param.data