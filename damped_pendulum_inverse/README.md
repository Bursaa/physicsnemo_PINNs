# Damped Pendulum Inverse Problem - Parameter Inference (Level 3)

**Complexity:** ⭐⭐⭐ ADVANCED  
**Equation:** `θ'' + ζ·θ' + (g/l)·sin(θ) = 0` (ζ unknown)  
**System Type:** Inverse problem - learn unknown parameter from observations

---

## 📖 Problem Description

### Physics
This is the most advanced problem: **learn unknown system parameters from noisy observations**.

**Governing Equation:**
$$\frac{d^2\theta}{dt^2} + \zeta\frac{d\theta}{dt} + \frac{g}{l}\sin(\theta) = 0$$

Where:
- **θ(t):** Angular displacement (observed, noisy)
- **ζ (zeta):** Damping ratio (**UNKNOWN** - must be inferred!)
- **g/l:** Gravitational coefficient (9.81 m/s²)

### The Inverse Problem
**Given:**
- Noisy observations: θ_obs(t) = θ_true(t) + ε

**Find:**
- Unknown parameter: ζ ∈ [0, 5]
- True trajectory: θ(t)

**Constraints:**
- Physics: ODE residual must be small
- Data fidelity: Prediction must match observations
- Smoothness: Solution should be regular

### Initial Conditions
- **θ(0) = π/3 rad** (60° initial angle)
- **dθ/dt(0) = 0 rad/s** (released from rest)
- **ζ_true = 0.5** (ground truth for validation)

### Simulation Parameters
| Parameter | Value | Meaning |
|-----------|-------|---------|
| Unknown damping (ζ) | 0.5 | Need to infer this! |
| Initial angle (θ₀) | π/3 (60°) | Moderate nonlinearity |
| Simulation time | 10 seconds | ~3-4 oscillations with decay |
| Observation noise | 1-5% Gaussian | Realistic measurement error |
| Observation points | 100-200 | Sparse sampling |

---

## 🧠 Dual-Network Architecture

### Two Neural Networks

#### Network 1: Forward Solution Network
```
Input: time t_norm ∈ [0,1]
    ↓
256 neurons × 6 layers (tanh)
    ↓
Output: angle θ(t)
```

**Purpose:** Predicts trajectory θ(t)

#### Network 2: Parameter Network
```
Input: 1 (constant, shared encoding)
    ↓
128 neurons × 3 layers (tanh)
    ↓
Output: scalar ζ (damping ratio)
```

**Purpose:** Predicts unknown parameter ζ

### Total Parameters
- Forward network: ~145,000
- Parameter network: ~15,000
- **Total:** ~160,000 (modest, joint training)

---

## ⚙️ Two-Phase Training Strategy

### Phase 1: Encoder Network (Initial Conditions)
**Purpose:** Learn which initial conditions best match observed data
**Time:** ~20 minutes
**Loss:** Data fidelity only (no physics yet)

```
Domain: Auxiliary encoder network
Input: Partial trajectory
Output: Predicted IC (θ₀, dθ/dt|₀)
Loss: L_encoder = ||θ_pred - θ_obs||²
```

**Output:** Pretrained IC predictor weights

### Phase 2: Inverse Problem Solver
**Purpose:** Learn both trajectory AND unknown parameter
**Time:** ~40 minutes
**Loss:** Physics + data constraints

```
Domain: Forward + Parameter networks
Inputs: Time t
Outputs: θ(t), ζ
Constraints:
  1. ODE residual: ||θ'' + ζθ' + (g/l)sin(θ)||
  2. Data fit: ||θ_pred - θ_obs||²
  3. Initial conditions: θ(0) = θ₀, θ'(0) = 0
Loss: L_total = w_ODE·L_ODE + w_data·L_data + w_IC·L_IC
```

**Output:** Trained forward network + parameter network

---

## 🔧 How to Run

### Setup
```bash
cd damped_pendulum_inverse/
conda activate physicsnemo
```

### Phase 1: Train Encoder
```bash
# Learn initial conditions from full trajectory
python damped_pendulum_encoder_solver.py

# Output: outputs/damped_pendulum_encoder_solver/
# Time: ~20 minutes
```

### Phase 2: Inverse Problem
```bash
# Learn both trajectory and unknown ζ
python damped_pendulum_inverse_solver.py

# Output: outputs/damped_pendulum_inverse_solver/
# Time: ~40 minutes
# Total: ~1 hour
```

### Visualization
```bash
# Phase 1 results
python plot_results_encoder.py
# Shows: Predicted vs actual IC, encoder reconstruction

# Phase 2 results
python plot_results_inverse.py
# Shows: Inferred ζ, trajectory comparison, phase space
```

### Configuration Override
```bash
# Different observation noise level (Phase 2)
python damped_pendulum_inverse_solver.py \
  noise_level=0.05 \
  training.max_steps=75000

# Different network size
python damped_pendulum_inverse_solver.py \
  arch.fully_connected.layer_size=512
```

---

## 📊 Expected Results

### Phase 1: Encoder Accuracy
```
✓ Predicted IC close to actual: θ₀_pred = θ₀_true ± 0.05 rad
✓ Encoder trajectory reconstruction: RMSE < 0.1 rad
✓ Training loss converges smoothly
```

### Phase 2: Parameter Inference
```
✓ ζ inference: ζ_pred = ζ_true ± 0.05 (1% noise)
✓ ζ inference: ζ_pred = ζ_true ± 0.10 (5% noise)
✓ Trajectory RMSE: 0.05-0.15 rad
✓ Phase error: < ±5°
```

### Accuracy by Noise Level
| Noise | ζ Error | θ RMSE | Difficulty |
|-------|---------|--------|-----------|
| 0% (synthetic) | ±0.01 | 0.02 | Easiest |
| 1% | ±0.03 | 0.05 | Easy |
| 5% | ±0.08 | 0.12 | Medium |
| 10% | ±0.15 | 0.25 | Hard |

---

## 🎯 Inverse Problem Challenges

### Challenge 1: Non-Identifiability
**Problem:** Different (θ, ζ) pairs may fit same data equally well

**Example:**
- (θ with ζ=0.3, more oscillations) vs
- (θ with ζ=0.5, fewer oscillations)

Both could match sparse noisy observations!

**Solution:**
- Use physics constraints (ODE residual)
- Add regularization on ζ
- Use long time horizon (more oscillations)
- Multiple observation frequencies help

### Challenge 2: Local Minima
**Problem:** Optimization may get stuck in local minimum

**Solution:**
- Good initial guess for ζ (e.g., ζ=0.5)
- Multi-phase training (forward first)
- Higher learning rates
- Larger networks

### Challenge 3: Noise Sensitivity
**Problem:** Parameter inference sensitive to measurement noise

**Solution:**
- Smooth observations (moving average filter)
- Use more observation points
- Add regularization term
- Uncertainty quantification

### Challenge 4: Multiple Parameters
**Problem:** If multiple unknowns (not just ζ), problem becomes harder

**Fundamental limit:** Number of observations must exceed number of unknowns

---

## 🔴 Troubleshooting

### Problem: ζ Inference Way Off
**Symptoms:** ζ_pred ≠ ζ_true (e.g., 0.1 vs 0.5)

**Root Cause:** Non-identifiability, poor initial guess, or noisy data

**Solutions:**
1. Ensure observation data is smooth (filter noise)
2. Provide better initial guess: `zeta_init=0.5`
3. Use more observation points (100 → 300)
4. Add regularization: `lambda_zeta=1000`
5. Longer simulation time (10s → 20s)
6. Higher learning rate (0.001 → 0.005)

### Problem: Phase 1 Encoder Doesn't Converge
**Symptoms:** IC prediction error doesn't decrease

**Root Cause:** Encoder network too small or learning rate too low

**Solutions:**
1. Increase network size: `layer_size: 256 → 512`
2. Increase encoder steps: `max_steps: 30000 → 50000`
3. Higher learning rate: `lr: 0.001 → 0.003`
4. Use more IC training points
5. Verify encoder uses pretrained Phase 1 weights

### Problem: Phase 2 Diverges After Phase 1
**Symptoms:** Loss increases after loading Phase 1 weights

**Root Cause:** Learning rate too high, weights not properly initialized

**Solutions:**
1. Lower learning rate: `lr: 0.001 → 0.0005`
2. Check encoder weights loaded correctly
3. Verify IC constraints from Phase 1 are applied
4. Use smaller learning rate decay
5. Frozen encoder weights first 5000 steps

### Problem: Trajectory Doesn't Match Observations
**Symptoms:** PINN trajectory deviates from data

**Root Cause:** Data weight too low or ODE weight too high

**Solutions:**
1. Increase data fidelity weight: `lambda_data: 100 → 1000`
2. Decrease ODE weight: `lambda_ODE: 1 → 0.1`
3. Check observation data is valid (no outliers)
4. Use observation uncertainty weighting
5. Longer training: `max_steps: 50000 → 75000`

---

## 📈 Understanding Output Files

### Training Outputs (Phase 1)
```
outputs/damped_pendulum_encoder_solver/
├── .hydra/
│   └── config_encoder.yaml         # Phase 1 configuration
├── checkpoints/
│   ├── epoch_0000.pt               # Encoder checkpoints
│   └── epoch_0060.pt (final)
├── damped_pendulum_encoder_solver_output.npz
│   ├── t: Time points
│   ├── theta_obs: Observed trajectory
│   ├── theta_pred: Encoder reconstruction
│   ├── IC_pred: Predicted initial conditions
│   └── loss_history: Training loss
└── training_logs.txt
```

### Training Outputs (Phase 2)
```
outputs/damped_pendulum_inverse_solver/
├── .hydra/
│   └── config.yaml                 # Phase 2 configuration
├── checkpoints/
│   ├── epoch_0000.pt               # Forward network
│   ├── epoch_0000_param.pt         # Parameter network
│   └── epoch_0100.pt (final)
├── damped_pendulum_inverse_solver_output.npz
│   ├── t: Time points
│   ├── theta_pred: Forward network prediction
│   ├── theta_obs: Observed trajectory
│   ├── zeta_pred: Inferred damping ratio
│   ├── zeta_std: Uncertainty (if available)
│   └── loss_components: IC, ODE, data losses
└── training_logs.txt
```

### Result Visualizations
```
outputs/damped_pendulum_encoder_solver/
├── encoder_ic_prediction.png       # Predicted vs actual IC
├── encoder_trajectory.png          # Reconstruction accuracy
└── encoder_convergence.png         # Loss over training

outputs/damped_pendulum_inverse_solver/
├── inferred_parameter.png          # ζ_pred vs ζ_true
├── trajectory_comparison.png       # PINN vs observations
├── phase_space.png                 # Phase portrait
├── parameter_convergence.png       # ζ evolution over training
└── loss_components.png             # ODE, data, IC losses
```

---

## 🔬 Physics Insights

### Damping Identification
The damping ratio ζ affects:
- **Oscillation frequency:** ωd = ω₀√(1-ζ²) (lower with more damping)
- **Decay rate:** e^(-ζω₀t) (faster with more damping)
- **Quality factor:** Q = 1/(2ζ) (oscillations per decay time)

Network must learn these relationships from observation data.

### Observable vs Unobservable
**Observable:** ζ directly affects what we see
- Oscillation frequency changes
- Decay envelope changes
- Easy to infer from full trajectory

**Unobservable:** Some parameters don't affect dynamics
- Example: gravity in small-angle limit (sin(θ) ≈ θ)
- Need additional information to identify

**Our case:** ζ is highly observable!

### Why Two-Phase Training?

**Phase 1 benefits:**
- IC network provides good initial guess for θ(0), θ'(0)
- Easier to learn ICs separately before adding parameter inference
- Faster convergence in Phase 2

**Phase 2 benefits:**
- Uses Phase 1 weights as initialization
- Focuses on finding ζ with good trajectory guess
- Physics constraints (ODE) guide parameter learning

### Identifiability Condition
For unique ζ inference, need:
- Observation time > 2π/ωd (at least 1-2 complete periods)
- Enough observation points (>50)
- Low noise level (<5%)
- Initial angle moderate (nonlinear but not extreme)

---

## 📚 References

### Inverse Problems
- **Tarantola (2005):** "Inverse Problem Theory" (classical reference)
- **Kaipio & Somersalo (2005):** "Statistical and Computational Inverse Problems"
- **Vogel (2002):** "Computational Methods for Inverse Problems"

### PINN for Inverse Problems
- **Raissi et al. (2019):** Original PINN paper (includes inverse)
- **Jagtap et al. (2020):** "Conservative Physics-Informed Neural Networks"

### System Identification
- **Ljung (1999):** "System Identification: Theory for the User"
- **Soderstrom & Stoica (1989):** "System Identification"

---

## ✅ Validation Checklist

### Phase 1 (Encoder)
- [ ] Encoder loss converges below 0.01
- [ ] Predicted IC close to actual: error < 0.05 rad
- [ ] Encoder trajectory reconstruction: RMSE < 0.1 rad
- [ ] Both IC components learned (not just θ)

### Phase 2 (Inverse Problem)
- [ ] ODE residual small (< 1e-3)
- [ ] Data fidelity error < observation noise level
- [ ] ζ inference within ±0.05 of true value (1% noise)
- [ ] Phase space matches expected spiral
- [ ] Trajectory passes near observation points
- [ ] Parameter network output is smooth
- [ ] Results reproducible with different random seeds

---

## 💡 Tips for Practitioners

### Quick Start Strategy
1. Generate clean synthetic data with known ζ_true
2. Run Phase 1 (encoder) - verify IC prediction works
3. Run Phase 2 (inverse) - verify ζ inference
4. Add noise gradually (1% → 5% → 10%)
5. Finally try real noisy observations

### Debugging Workflow
1. Check if Phase 1 encoder works at all
2. Verify observation data shape and scale
3. Print ζ_pred every 100 steps to see convergence
4. Compare ODE, IC, and data loss components
5. Plot loss curves - should be smooth and decreasing

### Hyperparameter Tuning
- **ζ not converging?** Increase data weight, use longer horizon
- **Diverges after Phase 1?** Lower Phase 2 learning rate
- **Encoder doesn't learn ICs?** Use larger encoder network
- **Slow convergence?** Higher learning rate, more training steps

### Noise Handling
- **Clean synthetic data:** Should easily get ζ error < 0.01
- **1% noise:** Expect ζ error ±0.03-0.05
- **5% noise:** Expect ζ error ±0.08-0.10
- **10% noise:** Expect ζ error ±0.15-0.20 (approach limit)

### Multiple Parameters
If inferring ζ and ω₀ jointly:
- Much harder problem (more nonuniqueness)
- Need longer observations
- Parameter network needs more capacity
- Consider sequential identification (Phase 1 ζ, Phase 2 ω₀)

---

## 🎓 Learning Outcomes

After completing this project, you understand:

- ✓ Inverse problem formulation & parameter inference
- ✓ Non-identifiability challenges in inverse problems
- ✓ Multi-phase training strategies
- ✓ Hybrid physics + data constraints
- ✓ Dual-network architectures
- ✓ Real-world system identification workflow
- ✓ Noise sensitivity and regularization
- ✓ Practical debugging techniques

---

**Status:** ✓ Complete implementation with two-phase training  
**Last Updated:** January 28, 2026  
**Difficulty:** Advanced (integrate concepts from Levels 1 & 2)

---

## 🚀 Recommended Project Extensions

1. **Multiple unknowns:** Infer ζ AND ω₀ simultaneously
2. **Structured uncertainty:** Use Bayesian PINN for confidence intervals
3. **Real data:** Try with actual pendulum measurements
4. **Time-varying parameters:** Learn ζ(t) that changes over time
5. **Coupled systems:** Two pendulums with unknown coupling
6. **Hybrid models:** Some terms known (gravity), others unknown (friction)
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
