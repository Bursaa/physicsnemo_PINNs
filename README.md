# PhysicsNEMO-BN: Physics-Informed Neural Networks for Pendulum Systems

**Framework:** NVIDIA PhysicsNEMO Symbolic (physicsnemo-sym)  
**Purpose:** Benchmark Physics-Informed Neural Networks (PINNs) for pendulum dynamics  
**Status:** Complete Implementation with Production-Ready Code

---

## 🎯 Quick Start (Choose Your Path)

### I'm New to PINNs
→ **[pendulum_ode/](pendulum_ode/)** - Start here!
- Simplest system: θ'' + (g/l)·sin(θ) = 0
- Learn PINN basics and supervised learning
- Training time: 15-20 minutes

### I Understand Basic PINNs
→ **[damped_pendulum/](damped_pendulum/)** - Next level
- Multiple forces: damping + elasticity + gravity
- Non-conservative system with energy dissipation
- Training time: 30-40 minutes

### I Want Advanced Techniques
→ **[damped_pendulum_inverse/](damped_pendulum_inverse/)** - Expert level
- Learn unknown parameters from observations
- Two-phase training strategy
- Training time: 1-1.5 hours

---

## 🏗️ Project Overview

This repository implements **three Physics-Informed Neural Networks** of increasing complexity:

| System | Equation | Key Features | Difficulty |
|--------|----------|--------------|------------|
| **Pendulum ODE** | `θ'' + (g/l)·sin(θ) = 0` | Hamiltonian, energy conservation, supervised learning | ⭐ Beginner |
| **Damped Pendulum** | `θ'' + 2ζω₀·θ' + ω₀²·θ + (g/l)·sin(θ) = 0` | Multi-physics, decay, complex dynamics | ⭐⭐ Intermediate |
| **Inverse Problem** | `θ'' + ζ·θ' + (g/l)·sin(θ) = 0` (ζ unknown) | Parameter inference, dual networks | ⭐⭐⭐ Advanced |

---

## 📚 Project Structure

```
physicsNEMO_BN/
│
├── README.md                           ← YOU ARE HERE
├── environment.yml                     ← Conda environment
├── FILE_INDEX.md                       ← Complete file reference
│
├── pendulum_ode/                       ← LEVEL 1: Basic PINN
│   ├── README.md                       ← Full documentation
│   ├── pendulum_ode.py                 ← PDE definition
│   ├── pendulum_solver.py              ← Training code
│   ├── plot_results_pendulum.py        ← Visualization
│   ├── conf/config.yaml                ← Hyperparameters
│   └── outputs/                        ← Results
│
├── damped_pendulum/                    ← LEVEL 2: Multi-Physics
│   ├── README.md                       ← Full documentation
│   ├── damped_pendulum.py              ← PDE definition
│   ├── damped_pendulum_solver.py       ← Training code
│   ├── plot_results_damped.py          ← Visualization
│   ├── conf/config.yaml                ← Hyperparameters
│   └── outputs/                        ← Results
│
└── damped_pendulum_inverse/            ← LEVEL 3: Inverse Problem
    ├── README.md                       ← Full documentation
    ├── damped_pendulum_inverse.py      ← PDE definition
    ├── damped_pendulum_encoder_solver.py    ← Phase 1
    ├── damped_pendulum_inverse_solver.py    ← Phase 2
    ├── plot_results_encoder.py         ← Phase 1 viz
    ├── plot_results_inverse.py         ← Phase 2 viz
    ├── conf/
    │   ├── config_encoder.yaml         ← Phase 1 params
    │   └── config.yaml                 ← Phase 2 params
    └── outputs/                        ← Results
```

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- CUDA 12.0+ (GPU strongly recommended)
- GCC 9+ (for compiling)
- Linux/Unix system

### Setup
```bash
# 1. Activate GCC (CentOS 8)
scl enable gcc-toolset-9 bash

# 2. Create conda environment
conda env create -f environment.yml
conda activate physicsnemo

# 3. Verify installation
python -c "import physicsnemo; import torch; print('✓ Ready')"
```

---

## ⚡ Running Experiments

### Level 1: Simple Pendulum (15 min)
```bash
cd pendulum_ode
python pendulum_solver.py          # Train
python plot_results_pendulum.py    # Visualize
```

### Level 2: Damped Pendulum (35 min)
```bash
cd damped_pendulum
python damped_pendulum_solver.py   # Train
python plot_results_damped.py      # Visualize
```

### Level 3: Inverse Problem (60 min)
```bash
cd damped_pendulum_inverse

# Phase 1: Learn initial conditions
python damped_pendulum_encoder_solver.py

# Phase 2: Infer unknown parameter ζ
python damped_pendulum_inverse_solver.py

# Visualize both phases
python plot_results_encoder.py
python plot_results_inverse.py
```

---

## 📖 Documentation

### Main Documentation
- **[FILE_INDEX.md](FILE_INDEX.md)** - Complete file reference & navigation map
- **[pendulum_ode/README.md](pendulum_ode/README.md)** - Level 1 detailed guide
- **[damped_pendulum/README.md](damped_pendulum/README.md)** - Level 2 detailed guide  
- **[damped_pendulum_inverse/README.md](damped_pendulum_inverse/README.md)** - Level 3 detailed guide

### What You'll Learn

#### From Level 1
- ✓ PINN basics: combining neural networks with physics constraints
- ✓ Why supervised learning prevents collapse to trivial solutions
- ✓ Energy conservation in Hamiltonian systems
- ✓ Initial condition constraints
- ✓ Basic debugging techniques

#### From Level 2
- ✓ Multi-physics systems (damping + elasticity + gravity)
- ✓ Non-conservative systems and energy dissipation
- ✓ Complex dynamics and multiple timescales
- ✓ Larger, more sophisticated networks
- ✓ Decay envelope validation

#### From Level 3
- ✓ Parameter inference from observations
- ✓ Two-phase training strategies
- ✓ Dealing with non-identifiability
- ✓ Dual-network architectures
- ✓ Real-world inverse problems

---

## 🔧 Configuration & Hyperparameters

Each system has a configuration file `conf/config.yaml`:

```yaml
arch:
  fully_connected:
    layer_size: 128         # Hidden units per layer
    nr_layers: 8            # Number of hidden layers

optimizer:
  lr: 0.002                 # Learning rate
  
scheduler:
  decay_rate: 0.95          # LR decay factor
  decay_steps: 1500         # Decay frequency

training:
  max_steps: 25000          # Training iterations

batch_size:
  IC: 200                   # Initial condition points
  interior: 1000            # Interior constraint points
```

### Parameter Tuning Guide

| Problem | Solution |
|---------|----------|
| Slow convergence | Increase `lr` (0.002 → 0.005) |
| Network too small | Add layers/units (`nr_layers: 8 → 10`) |
| GPU memory error | Reduce network size |
| Wrong frequency | Use supervised learning from reference solver |
| Phase drift | Increase interior batch size |

---

## 🧪 Key Innovation: Supervised Learning

The **critical insight** that makes these PINNs work is combining physics constraints with supervised learning:

```python
# Instead of pure PINN (physics only)
L_total = λ₁·L_IC + λ₂·L_ODE

# We use hybrid approach (physics + data)
L_total = λ₁·L_IC + λ₂·L_ODE + λ₃·L_supervised

# Where L_supervised comes from a high-precision reference solver
```

This solves the **collapse problem**: pure physics constraints allow θ=0 as valid solution. Supervised points force network to learn oscillatory behavior.

---

## 📊 Expected Results

### Pendulum ODE
```
✓ Amplitude preserved: π/6 ± 0.01 rad
✓ Period accuracy: 2.0 ± 0.05 s
✓ RMSE vs reference: 0.03-0.05 rad
✓ Training time: 15-20 min
```

### Damped Pendulum
```
✓ Decay: Exponential with correct envelope
✓ Phase space: Inward spiral
✓ RMSE vs reference: 0.05-0.10 rad  
✓ Training time: 30-40 min
```

### Inverse Problem
```
✓ ζ inference error: ±0.02-0.05
✓ Trajectory RMSE: 0.05-0.15 rad
✓ Phase 1 time: 20 min
✓ Phase 2 time: 35-45 min
```

---

## 🧠 Physics Background

### Mathematical Pendulum
- Simple nonlinear ODE from Lagrangian mechanics
- Exhibits periodic motion with amplitude-dependent period
- Energy conserving (Hamiltonian system)
- Known exact solutions in terms of elliptic integrals

### Damped Pendulum
- Combines spring restoring force, damping, and gravity
- Three damping regimes: underdamped, critically damped, overdamped
- Energy monotonically decreases
- Multiple timescales to learn

### Inverse Problems
- Learn unknown system parameter (damping ratio ζ) from observations
- Non-unique solutions possible (identifiability challenge)
- Requires careful regularization
- Real-world application for system identification

---

## 🔴 Troubleshooting

### Network Collapses to θ=0
**Fix:** 
- Increase ODE constraint weight
- Add supervised learning from reference solution
- Use larger network (128+ units)
- Train longer (40,000+ steps)

### Phase Drift (Errors Accumulate)
**Fix:**
- Increase interior constraint points
- Add Hamiltonian energy constraint
- Use periodic boundary conditions
- Longer training with lower decay rate

### Wrong Oscillation Frequency
**Fix:**
- Verify reference ODE solver is accurate
- Increase supervised constraint weight
- Use larger network
- Check time normalization

### Poor Inverse Problem Convergence
**Fix:**
- Good initial guess for parameter ζ
- Ensure observation data is clean/smooth
- Use multi-phase training (forward then inverse)
- Add regularization

See system-specific README files for detailed troubleshooting.

---

## 📚 Key Files Explained

### Physics Definition (`*_ode.py`)
Defines differential equations using SymPy:
```python
class MathematicalPendulum(PDE):
    def __init__(self, ...):
        self.equations["ode_pendulum"] = theta.diff(t, 2) + (g/l)*sin(theta)
```

### Training Script (`*_solver.py`)
Implements PINN training loop with:
- Domain setup with constraints
- Reference ODE solution generation
- Supervised learning from reference
- Validation and checkpointing
- Result visualization

### Visualization (`plot_results_*.py`)
Generates comparison plots:
- Trajectory: PINN vs reference
- Phase portrait
- Error analysis
- Energy conservation
- Parameter inference (inverse only)

### Configuration (`conf/config.yaml`)
Hyperparameters for entire experiment:
- Network architecture
- Optimizer settings
- Training parameters
- Batch sizes and loss weights

---

## 🤝 Contributing & Extensions

### To Extend This Project

1. **New pendulum variant:** Create `<system>_ode.py` with new equations
2. **New training technique:** Modify `*_solver.py` constraints
3. **Different network architecture:** Edit `conf/config.yaml`
4. **Coupled systems:** Extend to multiple coupled pendulums

### To Report Issues

1. Identify which system shows problem
2. Check system-specific README troubleshooting
3. Adjust configuration and retry
4. Include training loss curves in report

---

## 📖 Reference Materials

### PINN Literature
- **Raissi et al. (2019):** "Physics-informed neural networks: A deep learning framework"
- **Han et al. (2018):** "Solving high-dimensional PDEs using deep learning"
- **Cuomo et al. (2022):** "Scientific Machine Learning through PINNs"

### Physics References
- **Goldstein (2002):** Classical Mechanics, Chapter 3 (Pendulum)
- **Strogatz (2018):** Nonlinear Dynamics and Chaos, Chapter 4
- **Nayfeh & Balachandran (1995):** Applied Nonlinear Dynamics

### Framework Documentation
- [NVIDIA Modulus Docs](https://docs.nvidia.com/deeplearning/modulus/index.html)
- [GitHub Repository](https://github.com/NVIDIA/modulus-sym)

---

## ✅ Validation Checklist

Before accepting results:

- [ ] Code runs without errors
- [ ] Training loss converges (decreasing smoothly)
- [ ] Output plots look physically reasonable
- [ ] IC constraints satisfied to high precision
- [ ] ODE constraints satisfied (residual small)
- [ ] Energy/decay behavior correct
- [ ] RMSE vs reference within target
- [ ] Results reproducible with same random seed

---

## 📞 Quick Help

| Need | Location |
|------|----------|
| How to install? | [Installation section above](#-installation) |
| How to run experiment? | [Running Experiments section](#-running-experiments) |
| Understanding system X? | [System-specific README](pendulum_ode/README.md) |
| All files explained? | [FILE_INDEX.md](FILE_INDEX.md) |
| Configuration options? | [Configuration section](#-configuration--hyperparameters) |
| Fixing problem X? | System's README → Troubleshooting |

---

## 📋 Implementation Summary

### Architecture Progression
- **Level 1:** 128×8 (simple, ~15 min training)
- **Level 2:** 256×8 (intermediate, ~35 min training)
- **Level 3:** Dual 128×6 + 128×3 (advanced, ~60 min training)

### Key Technical Decisions
- **Tanh activation:** Better for periodic functions than ReLU
- **Supervised learning:** Prevents collapse to trivial solution
- **Normalized time:** Improves long-horizon training (t_norm = t/t_max)
- **Multi-phase training:** Phase 1 learns ICs, Phase 2 learns parameters

### Loss Function Strategy
```
L_total = w_IC·L_IC + w_ODE·L_ODE + w_supervised·L_supervised
        = 10000·L_IC + 5000·L_ODE + 15000·L_supervised
```

The high supervised weight (15,000) dominates early training to establish oscillatory behavior, while lower weights prevent over-fitting to noise.

---

## 🎓 Learning Path

**Recommended progression (4-7 hours):**

1. **Read this file** (10 min)
2. **Install & test** (20 min)
3. **Level 1:** Read guide, train, analyze (1-2 hours)
4. **Level 2:** Read guide, train, analyze (1-1.5 hours)
5. **Level 3:** Read guide, train both phases, analyze (1.5-2 hours)
6. **Experiment:** Modify hyperparameters and observe effects (flexible)

---

## 🚀 Next Steps

1. **Install environment** (if not done):
   ```bash
   conda env create -f environment.yml
   conda activate physicsnemo
   ```

2. **Run simplest experiment**:
   ```bash
   cd pendulum_ode
   python pendulum_solver.py
   python plot_results_pendulum.py
   ```

3. **Read system-specific README** in same folder

4. **Examine outputs** in `outputs/pendulum_solver/`

5. **Try your modifications** in `conf/config.yaml`

---

**Last Updated:** January 28, 2026  
**Framework:** NVIDIA PhysicsNEMO Symbolic  
**Status:** Production-Ready with Complete Documentation  