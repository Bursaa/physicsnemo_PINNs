# Damped Elastic Pendulum PINN

Physics-Informed Neural Network solving the damped elastic pendulum differential equation.

## Equation

$$\frac{d^2\theta}{dt^2} + 2\zeta\omega_0 \frac{d\theta}{dt} + \omega_0^2 \theta + \frac{g}{l}\sin(\theta) = 0$$

Where:
- **θ(t)**: angle from vertical (radians)
- **ζ (zeta)**: damping ratio (0.3 in this example → underdamped)
- **ω₀**: natural frequency of the system (2.0 rad/s)
- **g/l**: gravitational restoring coefficient (9.81 m/s²)

## Physical Interpretation

This model combines:
1. **Spring mechanics**: `ω₀²·θ` (elastic restoring force)
2. **Damping**: `2ζω₀·dθ/dt` (friction/energy dissipation)
3. **Pendulum gravity**: `(g/l)·sin(θ)` (gravitational torque)

The system exhibits **exponentially decaying oscillations** due to damping.

## Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| ζ | 0.3 | Underdamped (oscillates while decaying) |
| ω₀ | 2.0 rad/s | Natural frequency |
| θ₀ | 0.3 rad | Initial angle |
| θ'₀ | 0.0 rad/s | Initial velocity (released from rest) |
| t_max | 15.0 s | Simulation time |

## Quick Start

```bash
# Run training (takes ~30 min on GPU)
python damped_elastic_pendulum_solver.py

# Analyze results
python plot_results_damped.py

# Override configuration
python damped_elastic_pendulum_solver.py training.max_steps=20000 arch.fully_connected.layer_size=256
```

## Architecture

- **Input**: time `t`
- **Hidden layers**: 10 layers × 512 neurons each
- **Output**: angle `θ(t)` and automatic derivatives `θ'(t)`, `θ''(t)`
- **Physics constraint**: ODE residual must equal zero
- **Data constraint**: Initial conditions at t=0

## Key Differences from Simple Pendulum

- **Harder problem**: Combines multiple physical effects
- **Longer time horizon**: 15s instead of 10s (more decay to learn)
- **Non-linear damping**: Requires learning the decay envelope
- **More complex dynamics**: Underdamped oscillation with exponential decay

## Expected Results

| Metric | Target |
|--------|--------|
| Mean absolute error | < 0.001 rad |
| Max error | < 0.01 rad |
| Training time | ~30 min (GPU) |
| Amplitude decay | Exponential (correct) |

## Troubleshooting

**Error: "Could not sample interior of geometry"**
- The ODE constraint uses `PointwiseConstraint.from_numpy` with explicit time sampling
- This avoids geometry issues

**Poor accuracy after training**
- Increase `training.max_steps` to 200000
- Increase `batch_size.interior` to 10000
- Try smaller learning rate: `optimizer.lr=0.00005`

**Slow training**
- Decrease `training.max_steps` to 50000
- Decrease `batch_size` parameters by half
- Use smaller architecture: `layer_size=256 nr_layers=8`
