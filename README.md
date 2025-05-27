# Physics-Informed Neural Networks for the Lorenz Attractor
## A Differentiable Programming Approach

## Project Overview

This repository contains the implementation of Physics-Informed Neural Networks (PINNs) applied to modeling the Lorenz attractor system. The project demonstrates the application of differentiable programming techniques to solve partial differential equations using deep learning, specifically focusing on the chaotic dynamics of the Lorenz system.

### Key Features

- Complete PINN implementation from scratch using PyTorch
- Analysis of the Lorenz attractor dynamics
- Differentiable programming techniques for scientific computing
- Visualization and analysis tools

## Installation and Setup

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/noahnovsak/diffprog-project.git
cd diffprog-project

# Create virtual environment using venv/pip
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Or using uv
uv sync
source .venv/bin/activate
```

### Physics-Informed Neural Networks (PINNs)

The core PINN implementation incorporates:
- Automatic differentiation for computing derivatives
- Physics-based loss functions encoding the Lorenz equations
- Initial condition constraints
- Advanced techniques for training

### Lorenz Attractor System

The Lorenz system is defined by:
```
dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y
dz/dt = xy - βz
```

Where σ, ρ, and β are the system parameters.
