# Physics-Informed Neural Networks for the Lorenz Attractor

This repository contains the implementation of Physics-Informed Neural Networks (PINNs) for modeling the chaotic Lorenz attractor system. The code demonstrates various architectural enhancements and training strategies to achieve stable long-term prediction of chaotic dynamics.

## Overview

Physics-Informed Neural Networks embed governing physical laws directly into neural network loss functions, enabling equation-guided learning without requiring large labeled datasets. This project specifically tackles the challenging Lorenz system, known for its sensitive dependence on initial conditions and chaotic behavior.

## Features

- **Multiple PINN Architectures**: Standard MLP and enhanced MLP with skip connections
- **Fourier Embeddings**: High-frequency signal modeling for complex dynamics
- **Causal Loss Functions**: Prioritizing temporal causality to prevent trivial solutions
- **Domain Splitting**: Training strategy for improved stability over long time horizons
- **Evaluation**: Relative L2 error metrics and visualization tools

## Repository Structure

```
src/
├── model.py                # Neural network backbone architectures (MLP, Modified MLP, Fourier embeddings)
├── pinn.py                 # Lorenz attractor PINN implementation and physics constraints
├── train.py                # Training loop, evaluation functions
├── utils.py                # Utility functions
├── notebooks/
│   ├── training.ipynb      # Main training notebook - experiments and model comparisons
│   └── visualization.ipynb # Results visualization and figure generation
├── figures/                # Generated plots and animations
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- NumPy
- Matplotlib
- SciPy
- Jupyter Notebook

### Setup
```bash
git clone https://github.com/noahnovsak/diffprog-project.git
cd diffprog-project
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Or using uv
uv sync
source .venv/bin/activate
```

### Running the Project
1. **Open `training.ipynb`** to run the main experiments:
   - Model architecture comparisons
   - Training with different loss functions
   - Domain splitting demonstrations
   - Performance evaluations

2. **Open `visualization.ipynb`** to generate plots and visualizations:
   - Trajectory comparisons
   - Training loss curves  
   - Model performance analysis
   - Animation generation

3. **For script-based usage**, import the modules directly:
   ```python
   from model import MLP, ModifiedMLP, FourierEmbedding
   from pinn import LorenzAttractor
   from train import train, eval
   ```

## Quick Start

### Basic Training
```python
from pinn import LorenzAttractor
from train import train

# Initial conditions and time range
ics = [1., 1., 1.]
t0, t1 = 0., .5

# Initialize Lorenz PINN
pinn = LorenzAttractor(ics, t0, t1)

# Train the model
trained_model = train(pinn, epochs=10000)
```

### Enhanced Training
```python
# Enhanced model with skip connections
modified_pinn = LorenzAttractor(ics, t0, t1, mlp='ModifiedMLP')

# Enhanced with Fourier embeddings
fourier_pinn = LorenzAttractor(ics, t0, t1, embed={"embed_dim": 256}, layers=[256, 256, 256, 3])
```

## Results

Key findings from our experiments:

- **Skip connections** improve convergence speed and stability
- **Fourier embeddings** enable better high-frequency modeling
- **Causal loss** prevents convergence to trivial solutions
- **Domain splitting** is most effective for long-term predictions
- Computational cost remains the primary limitation
- Training convergence can be sensitive to initialization
- Long-term predictions may accumulate errors despite PINNs formulation
- Google Colab session limits affect extended training runs
