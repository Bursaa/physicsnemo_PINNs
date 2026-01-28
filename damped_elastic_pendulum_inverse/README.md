# Encoder-Based Inverse PINN for Damped Pendulum Parameter Estimation

This module implements an **Encoder-Based Inverse Physics-Informed Neural Network (IPINN)** for estimating physical parameters of a damped pendulum from observational data. The model trains on **multiple cases** with different initial conditions and damping coefficients, enabling generalization to new, unseen time series.

## Problem Description

The damped pendulum is governed by the ODE:

$$\frac{d^2\theta}{dt^2} + \zeta \frac{d\theta}{dt} + \frac{g}{l}\sin(\theta) = 0$$

**Inverse Problem:** Given observations θ(t), estimate:
- **ζ (zeta):** Damping coefficient
- **θ₀:** Initial angle
- **ω₀:** Initial angular velocity

## Encoder-Based Inverse Solver

The main solver (`damped_elastic_pendulum_encoder_solver.py`) uses an encoder network to process time series and predict parameters. This enables true generalization to completely new cases.

```bash
python damped_elastic_pendulum_encoder_solver.py
```

**Architecture:**
1. **Encoder:** Processes time series [t, θ(t)] → embedding vector
2. **Parameter Decoder:** embedding → (ζ, θ₀, ω₀)
3. **Conditioned θ-Network:** (t, embedding) → θ(t)

**This approach can predict parameters for ANY new time series without retraining!**

## Configuration Files

| File | Description |
|------|-------------|
| `conf/config.yaml` | Single-case inverse problem (legacy) |
| `conf/config_encoder.yaml` | Encoder-based approach (recommended) |

### Key Configuration Parameters

```yaml
custom:
  n_training_cases: 100     # Number of training cases
  n_test_cases: 20          # Number of test cases
  n_time_points: 50         # Time points per case
  noise_level: 0.02         # Observation noise (std)
  
  lambda_data: 100.0        # Data fitting weight
  lambda_physics: 1.0       # ODE residual weight
  lambda_ic: 100.0          # Initial condition weight
```

## Visualization

```bash
# Encoder results
python plot_results_encoder.py
```

## Parameter Ranges

Default ranges for training data generation:

| Parameter | Range | Unit |
|-----------|-------|------|
| ζ (zeta) | [0.05, 0.8] | - |
| θ₀ | [-π/3, π/3] | rad |
| ω₀ | [-1.5, 1.5] | rad/s |

## Output Structure

```
outputs/
└── damped_elastic_pendulum_encoder_solver/
    ├── encoder_results.npz
    ├── encoder_results_plot.png
    └── model.pth
```

## Usage Example: Predicting Parameters for New Data

After training the encoder model, you can use it to predict parameters for new time series:

```python
import torch
import numpy as np
from damped_elastic_pendulum_encoder_solver import EncoderBasedParameterNet

# Load trained model
model = EncoderBasedParameterNet(n_time_points=100, embedding_dim=64, ...)
model.load_state_dict(torch.load("outputs/.../model.pth")['encoder_param_net'])
model.eval()

# Prepare your new time series data
t_new = np.linspace(0, 20, 100).reshape(1, -1, 1)  # (1, n_time_points, 1)
theta_new = your_observations.reshape(1, -1, 1)    # Your measured theta(t)

# Predict parameters
with torch.no_grad():
    output = model({
        "t_series": torch.tensor(t_new, dtype=torch.float32),
        "theta_series": torch.tensor(theta_new, dtype=torch.float32)
    })
    
print(f"Estimated ζ: {output['zeta'].item():.4f}")
print(f"Estimated θ₀: {output['theta0'].item():.4f}")
print(f"Estimated ω₀: {output['omega0'].item():.4f}")
```

## Loss Functions

The training minimizes a combined loss:

$$\mathcal{L} = \lambda_\text{data} \mathcal{L}_\text{data} + \lambda_\text{physics} \mathcal{L}_\text{ODE} + \lambda_\text{IC} \mathcal{L}_\text{IC}$$

Where:
- **Data loss:** $\mathcal{L}_\text{data} = \sum_i |\theta_\text{pred}(t_i) - \theta_\text{obs}(t_i)|^2$
- **Physics loss:** $\mathcal{L}_\text{ODE} = \sum_j |R(t_j)|^2$ (ODE residual)
- **IC loss:** $\mathcal{L}_\text{IC} = |\theta(0) - \theta_0|^2 + |\dot\theta(0) - \omega_0|^2$

## Tips for Best Results

1. **Noise sensitivity:** Increase `lambda_physics` if data is noisy
2. **Parameter ranges:** Ensure test cases fall within training ranges
3. **Training time:** Encoder approach needs more training steps (~30k+)
4. **Embedding dimension:** Larger embeddings capture more complex dynamics
5. **Time points:** More observation points improve parameter estimation
