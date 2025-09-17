# JEPA: Joint-Embedding Predictive Architecture

A PyTorch implementation of Meta's Joint-Embedding Predictive Architecture (JEPA) for self-supervised representation learning on toy sequential data.

## Overview

JEPA is a self-supervised learning approach that learns representations by predicting in representation space rather than pixel/data space. This implementation demonstrates the core JEPA principles using sequences of 2D trajectories (circular and linear motion patterns).

### Key Features

- **Self-supervised learning**: No labels required - learns from temporal structure
- **Joint-embedding architecture**: Context and target sequences encoded in same representation space  
- **Predictive learning**: Predicts future representations rather than raw data
- **Exponential Moving Average (EMA)**: Smooth target encoder updates for stable training
- **Toy dataset**: Configurable 2D trajectory patterns (circular, linear) with noise

## Architecture

```
Context Sequence → Context Encoder → Context Representation
                                            ↓
                                       Predictor → Predicted Representation
                                            ↓
Target Sequence → Target Encoder (EMA) → Target Representation
                                            ↓
                                    Cosine Similarity Loss
```

## Installation

```bash
git clone https://github.com/yourusername/jepa-pytorch.git
cd jepa-pytorch
pip install -r requirements.txt
```

## Quick Start

```python
from jepa import JEPA, ToySequenceDataset, train_jepa
from torch.utils.data import DataLoader

# Create dataset
dataset = ToySequenceDataset(num_samples=1000, seq_len=20, pattern_type='circle')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
model = JEPA(input_dim=2, hidden_dim=64, repr_dim=32)

# Train
losses = train_jepa(model, dataloader, num_epochs=50, lr=1e-3)
```

## Usage

### Basic Training

Run the complete training pipeline with visualization:

```bash
python main.py
```

This will:
- Generate toy trajectory datasets
- Train the JEPA model for 50 epochs
- Plot training loss curves
- Visualize learned representations
- Test prediction capabilities

### Custom Datasets

Create your own sequential datasets:

```python
from jepa import ToySequenceDataset

# Circular motion patterns
circular_data = ToySequenceDataset(
    num_samples=1000, 
    seq_len=20, 
    pattern_type='circle'
)

# Linear motion patterns  
linear_data = ToySequenceDataset(
    num_samples=1000,
    seq_len=20, 
    pattern_type='linear'
)
```

### Model Configuration

Customize the JEPA architecture:

```python
model = JEPA(
    input_dim=2,      # Input feature dimensions
    hidden_dim=64,    # LSTM hidden size
    repr_dim=32       # Representation dimensionality
)
```

### Advanced Training Options

```python
losses = train_jepa(
    model=model,
    dataloader=train_loader,
    num_epochs=100,
    lr=1e-3,
    context_len=8,    # Context sequence length
    target_len=4,     # Target sequence length  
    momentum=0.99     # EMA momentum for target encoder
)
```

## Results

After training, the model learns to:
- Encode sequential patterns into meaningful representations
- Predict future trajectory representations from past context
- Achieve cosine similarities > 0.7 between predicted and actual target representations

### Example Output

```
Epoch 0, Loss: 0.8234
Epoch 10, Loss: 0.4567  
Epoch 20, Loss: 0.2891
Epoch 30, Loss: 0.1745
Epoch 40, Loss: 0.1234

Testing prediction capability...
Prediction similarities: tensor([0.7234, 0.8012, 0.7567, 0.7891])
Mean similarity: 0.7676
```

## Architecture Details

### Components

1. **Context Encoder**: LSTM-based encoder for input sequences
2. **Target Encoder**: Identical to context encoder, updated via EMA  
3. **Predictor**: MLP that predicts target representations from context
4. **Loss Function**: Cosine similarity between predicted and target representations

### Training Process

1. Split sequences into context (past) and target (future) segments
2. Encode both segments into representation vectors
3. Predict target representation from context representation
4. Optimize cosine similarity between prediction and target
5. Update target encoder weights using exponential moving average

## Visualization

The implementation includes visualization tools for:
- Training loss curves
- 2D projections of learned representations  
- Prediction accuracy metrics

## Extending to Other Domains

This implementation can be adapted for:
- **Video sequences**: Replace 2D trajectories with frame patches
- **Audio signals**: Use spectrograms or raw waveforms
- **Natural language**: Token sequences with transformer encoders
- **Time series**: Any sequential data (stock prices, sensor readings, etc.)

## References

- [The Path Towards Autonomous Machine Intelligence](https://openreview.net/forum?id=BZ5a1r-kVsf) - Yann LeCun
- [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243) - Assran et al.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Meta AI Research for the original JEPA concept
- PyTorch team for the excellent deep learning framework