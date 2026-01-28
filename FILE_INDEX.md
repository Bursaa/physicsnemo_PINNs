# PhysicsNEMO-BN: Complete File Index & Documentation Map
## Every File Explained

**Generated:** January 28, 2026  
**Purpose:** Comprehensive reference for all files in the project

---

## 📖 Documentation Files (START HERE)

### Entry Points (Read in Order)

1. **[README.md](README.md)** - Original Installation Guide
   - **Purpose:** Setup environment on CentOS 8
   - **Audience:** System administrators, first-time setup
   - **Content:** GCC activation, conda environment, installation test
   - **Time to read:** 5 minutes
   - **When to use:** Initial setup only

2. **[README_QUICK_START.md](README_QUICK_START.md)** - Navigation & Quick Reference
   - **Purpose:** Choose your learning path through the project
   - **Audience:** Everyone (entry point)
   - **Content:** Quick start, folder structure, workflow
   - **Time to read:** 10 minutes
   - **When to use:** Before starting any experiment

3. **[README_LEGACY.md](README_LEGACY.md)** - Comprehensive Project Documentation
   - **Purpose:** Complete technical reference for entire project
   - **Audience:** Researchers, developers, advanced users
   - **Content:** 1500+ lines covering physics, architecture, troubleshooting
   - **Time to read:** 30-60 minutes
   - **When to use:** Deep dives, troubleshooting, theory

### System-Specific Guides

4. **[pendulum_ode/README_DETAILED.md](pendulum_ode/README_DETAILED.md)** - Level 1: Basic PINN
   - **Equation:** `θ'' + (g/l)·sin(θ) = 0`
   - **Content:** 400 lines of physics, architecture, troubleshooting
   - **Key topics:** PINN basics, supervised learning, collapse prevention
   - **Start here if:** You're new to PINNs

5. **[damped_pendulum/README_DETAILED.md](damped_pendulum/README_DETAILED.md)** - Level 2: Multi-Physics
   - **Equation:** `θ'' + 2ζω₀·θ' + ω₀²·θ + (g/l)·sin(θ) = 0`
   - **Content:** 450 lines on complex systems
   - **Key topics:** Multiple forces, damping, decay behavior
   - **Start here if:** You understand basic PINNs

6. **[damped_pendulum_inverse/README_DETAILED.md](damped_pendulum_inverse/README_DETAILED.md)** - Level 3: Inverse Problems
   - **Equation:** `θ'' + ζ·θ' + (g/l)·sin(θ) = 0` (ζ unknown)
   - **Content:** 550 lines on parameter inference
   - **Key topics:** Unknown parameters, two-phase training, identifiability
   - **Start here if:** You want advanced techniques

---

## 🔬 Physics Definition Files

### Pendulum ODE
**File:** [pendulum_ode/pendulum_ode.py](pendulum_ode/pendulum_ode.py)
- **Purpose:** Defines mathematical pendulum equation in SymPy
- **Key Class:** `MathematicalPendulum(PDE)`
- **Equation:** `θ'' + (g/l)·sin(θ) = 0`
- **Parameters:**
  - `g`: Gravitational acceleration (default: 9.81 m/s²)
  - `l`: Pendulum length (default: 1.0 m)
- **Output:** SymPy equation object for PINN training
- **Lines:** ~70 lines
- **Dependencies:** sympy, physicsnemo.sym.eq.pde

### Damped Elastic Pendulum
**File:** [damped_pendulum/damped_pendulum.py](damped_pendulum/damped_pendulum.py)
- **Purpose:** Defines damped elastic pendulum system
- **Key Class:** `DampedElasticPendulum(PDE)`
- **Equation:** `θ'' + 2ζω₀·θ' + ω₀²·θ + (g/l)·sin(θ) = 0`
- **Parameters:**
  - `zeta`: Damping ratio (0=undamped, 1=critical, >1=overdamped)
  - `omega0`: Natural frequency (rad/s)
  - `g`: Gravitational acceleration
  - `l`: Pendulum length
- **Output:** SymPy equation with four force terms
- **Lines:** ~75 lines
- **Dependencies:** sympy, physicsnemo

### Damped Pendulum Inverse
**File:** [damped_pendulum_inverse/damped_pendulum_inverse.py](damped_pendulum_inverse/damped_pendulum_inverse.py)
- **Purpose:** Defines damped pendulum for inverse problem (ζ unknown)
- **Key Class:** `DampedPendulumInverse(PDE)`
- **Equation:** `θ'' + ζ·θ' + (g/l)·sin(θ) = 0` where ζ is Symbol
- **Key difference:** ζ is a Symbol (not fixed), will be learned!
- **Output:** Equation allowing ζ to vary
- **Lines:** ~65 lines
- **Dependencies:** sympy, physicsnemo

---

## 🎯 Solver & Training Files

### Pendulum ODE Solver
**File:** [pendulum_ode/pendulum_solver.py](pendulum_ode/pendulum_solver.py)
- **Purpose:** Main training script for simple pendulum PINN
- **Key Steps:**
  1. Define network architecture (instantiate_arch)
  2. Create domain with constraints (IC, ODE, supervised)
  3. Generate reference solution (solve_ivp)
  4. Add validator and run training
- **Output:** Trained model, checkpoints, results
- **Parameters in code:**
  - θ₀ = π/6 (30°)
  - ω₀ = 0
  - t_max = 10 seconds
  - Network: 128×8 (tanh)
  - Steps: 25,000
- **Lines:** ~200 lines
- **Runtime:** 15-20 minutes
- **Key functions:**
  - `run()`: Main training entry point
  - Integration with Hydra config system
  - Loss calculation and validation

### Damped Pendulum Solver
**File:** [damped_pendulum/damped_pendulum_solver.py](damped_pendulum/damped_pendulum_solver.py)
- **Purpose:** Training script for damped elasticpendulum
- **Complexity:** Similar to pendulum_solver.py but with:
  - Larger network (256×8)
  - More interior points (5000)
  - Energy dissipation monitoring
  - Multiple damping regimes support
- **Output:** Trained model, decay curves
- **Parameters:**
  - ζ = 0.3, ω₀ = 1.5 rad/s
  - t_max = 20 seconds
  - Steps: 50,000
- **Runtime:** 30-40 minutes
- **Lines:** ~180 lines

### Encoder Solver (Phase 1)
**File:** [damped_pendulum_inverse/damped_pendulum_encoder_solver.py](damped_pendulum_inverse/damped_pendulum_encoder_solver.py)
- **Purpose:** Phase 1 of inverse problem - predict initial conditions
- **Input:** Full trajectory observation
- **Output:** Trained encoder predicting θ(0), dθ/dt(0)
- **Architecture:** Encoder network + integrator
- **Steps:** 30,000
- **Runtime:** ~20 minutes
- **Key insight:** Learns which initial conditions best match data

### Inverse Solver (Phase 2)
**File:** [damped_pendulum_inverse/damped_pendulum_inverse_solver.py](damped_pendulum_inverse/damped_pendulum_inverse_solver.py)
- **Purpose:** Phase 2 - infer damping ζ and trajectory
- **Input:** Observation data, encoder pretrained weights
- **Output:** Inferred ζ, full trajectory
- **Architecture:** Forward network + parameter network (dual)
- **Loss:** Data fit + ODE residual + regularization
- **Steps:** 50,000
- **Runtime:** 35-45 minutes
- **Key innovation:** Two networks trained jointly

---

## 📊 Visualization & Analysis Files

### Pendulum ODE Plots
**File:** [pendulum_ode/plot_results_pendulum.py](pendulum_ode/plot_results_pendulum.py)
- **Purpose:** Generate comparison plots for simple pendulum
- **Inputs:** 
  - Trained PINN model
  - Reference ODE solution
  - Configuration
- **Outputs:**
  - `trajectory.png` - θ(t) vs reference
  - `phase_space.png` - Phase portrait
  - `error_analysis.png` - PINN vs reference
  - `comparison_*.png` - Stage-by-stage evolution
- **Key metrics:**
  - RMSE (root mean square error)
  - Max error
  - Phase error
  - Energy conservation
- **Lines:** ~150 lines

### Damped Pendulum Plots
**File:** [damped_pendulum/plot_results_damped.py](damped_pendulum/plot_results_damped.py)
- **Purpose:** Visualization for damped system
- **Outputs:**
  - Trajectory with decay envelope
  - Energy dissipation curve
  - Phase space spiral
  - Decay rate verification
- **Additional metrics:**
  - Energy loss rate
  - Decay timescale
  - Oscillation frequency

### Encoder Results Plots
**File:** [damped_pendulum_inverse/plot_results_encoder.py](damped_pendulum_inverse/plot_results_encoder.py)
- **Purpose:** Visualize Phase 1 encoder performance
- **Outputs:**
  - Predicted vs actual initial conditions
  - Encoder trajectory reconstruction
  - Loss curves
- **Lines:** ~100 lines

### Inverse Problem Plots
**File:** [damped_pendulum_inverse/plot_results_inverse.py](damped_pendulum_inverse/plot_results_inverse.py)
- **Purpose:** Visualize Phase 2 inverse problem results
- **Outputs:**
  - Inferred ζ value with uncertainty
  - Trajectory comparison
  - Phase space comparison
  - Parameter convergence history
  - ODE residual visualization
- **Lines:** ~150 lines

---

## ⚙️ Configuration Files

### Pendulum ODE Config
**File:** [pendulum_ode/conf/config.yaml](pendulum_ode/conf/config.yaml)
- **Purpose:** Hyperparameters for pendulum ODE PINN
- **Key settings:**
  ```yaml
  arch:
    fully_connected:
      layer_size: 128
      nr_layers: 8
  optimizer:
    lr: 0.002
  training:
    max_steps: 25000
  batch_size:
    IC: 200
    interior: 1000
  ```
- **Customizable:** All parameters tunable for different results
- **Lines:** 45 lines

### Damped Pendulum Config
**File:** [damped_pendulum/conf/config.yaml](damped_pendulum/conf/config.yaml)
- **Similar to above but:**
  - Larger network (256 units, 8 layers)
  - More training steps (50,000)
  - Larger batch sizes (IC: 500, interior: 5000)
  - Different learning rate schedule

### Inverse Problem Configs
**Files:**
- [damped_pendulum_inverse/conf/config_encoder.yaml](damped_pendulum_inverse/conf/config_encoder.yaml) - Phase 1
- [damped_pendulum_inverse/conf/config.yaml](damped_pendulum_inverse/conf/config.yaml) - Phase 2

- **Difference:** Phase 2 has additional loss weights (data, ODE, regularization)

---

## 📦 Output Directory Structure

### Pendulum ODE Results
```
outputs/pendulum_solver/
├── .hydra/
│   ├── config.yaml          # Exact configuration used
│   ├── hydra.yaml           # Hydra framework config
│   └── overrides.yaml       # Command-line overrides (if any)
├── checkpoints/
│   ├── epoch_0000.pt        # Checkpoint at step 0
│   ├── epoch_0010.pt        # Every 500 steps (~500×10)
│   └── epoch_0050.pt        # Final checkpoint
├── pendulum_solver_output.npz   # Results: t, theta_pred, theta_ref, etc.
├── training_logs.txt        # Loss history
└── *.png                    # Generated plots (trajectory, phase space, etc.)
```

### Damped Pendulum Results
```
outputs/damped_elastic_pendulum_solver/
├── .hydra/                  # Same structure as above
├── checkpoints/
├── damped_elastic_pendulum_solver_output.npz
├── training_logs.txt
└── *.png                    # Decay, energy, phase space plots
```

### Inverse Problem Results
```
outputs/damped_pendulum_encoder_solver/
└── [Phase 1 outputs as above]

outputs/damped_pendulum_inverse_solver/
├── .hydra/
├── checkpoints/
├── damped_pendulum_inverse_solver_output.npz
│   ├── t: Time points
│   ├── theta_pred: Predicted trajectory
│   ├── theta_obs: Observed trajectory (if available)
│   ├── zeta_inferred: Learned damping parameter
│   ├── zeta_std: Uncertainty estimate
│   └── loss_history: Training loss curves
└── *.png                    # Parameter, trajectory, phase space plots
```

---

## 🌍 External Dependencies (Not Modified)

### PhysicsNEMO Framework
**Location:** physicsnemo-sym/ (external pip package)
- **Role:** Core PINN framework
- **Key components used:**
  - `physicsnemo.sym.eq.pde`: PDE definition base class
  - `physicsnemo.sym.hydra`: Configuration management
  - `physicsnemo.sym.solver`: PINN training solver
  - `physicsnemo.sym.domain`: Domain and constraint management
  - `physicsnemo.sym.models`: Neural network architectures
- **Installation:** Comes with conda environment
- **Documentation:** NVIDIA official docs

### Standard Libraries
- **torch**: Deep learning (CUDA enabled)
- **numpy**: Numerical computing
- **scipy**: Numerical integration (solve_ivp)
- **matplotlib**: Plotting
- **hydra**: Configuration framework
- **sympy**: Symbolic math (PDE definition)

---

## 📋 Project Files Summary Table

| File | Type | Purpose | Lines | Language |
|------|------|---------|-------|----------|
| README.md | Doc | Installation | 50 | Markdown |
| README_QUICK_START.md | Doc | Navigation | 300 | Markdown |
| README_LEGACY.md | Doc | Complete ref | 1500 | Markdown |
| pendulum_ode/README_DETAILED.md | Doc | Level 1 guide | 450 | Markdown |
| damped_pendulum/README_DETAILED.md | Doc | Level 2 guide | 450 | Markdown |
| damped_pendulum_inverse/README_DETAILED.md | Doc | Level 3 guide | 550 | Markdown |
| environment.yml | Config | Dependencies | 50 | YAML |
| pendulum_ode/pendulum_ode.py | Physics | ODE definition | 70 | Python |
| damped_pendulum/damped_pendulum.py | Physics | ODE definition | 75 | Python |
| damped_pendulum_inverse/damped_pendulum_inverse.py | Physics | ODE definition | 65 | Python |
| pendulum_ode/pendulum_solver.py | Code | Training | 200 | Python |
| damped_pendulum/damped_pendulum_solver.py | Code | Training | 180 | Python |
| damped_pendulum_inverse/damped_pendulum_encoder_solver.py | Code | Training (P1) | 150 | Python |
| damped_pendulum_inverse/damped_pendulum_inverse_solver.py | Code | Training (P2) | 180 | Python |
| pendulum_ode/plot_results_pendulum.py | Code | Visualization | 150 | Python |
| damped_pendulum/plot_results_damped.py | Code | Visualization | 150 | Python |
| damped_pendulum_inverse/plot_results_encoder.py | Code | Visualization | 100 | Python |
| damped_pendulum_inverse/plot_results_inverse.py | Code | Visualization | 150 | Python |
| pendulum_ode/conf/config.yaml | Config | Hyperparameters | 45 | YAML |
| damped_pendulum/conf/config.yaml | Config | Hyperparameters | 45 | YAML |
| damped_pendulum_inverse/conf/config_encoder.yaml | Config | Phase 1 params | 45 | YAML |
| damped_pendulum_inverse/conf/config.yaml | Config | Phase 2 params | 50 | YAML |
| **TOTAL** | | | **~6000** | |

---

## 🔍 File Dependency Graph

```
Main Entry Point:
  ↓
[README_QUICK_START.md] (navigation)
  ├─→ [README.md] (installation)
  │
  └─→ [README_LEGACY.md] (detailed reference)
  
For Each System:
  [System/README_DETAILED.md]
    ├─→ [System/*_ode.py] (SymPy equation)
    ├─→ [System/*_solver.py] (training script)
    │   └─→ [System/conf/config.yaml] (hyperparameters)
    └─→ [System/plot_results_*.py] (visualization)
        └─→ [System/outputs/] (results)

External Dependencies:
  [environment.yml]
    └─→ [physicsnemo] (NVIDIA framework)
```

---

## 🚀 Typical User Journey Through Files

### Day 1: Installation & Quick Start
1. Read: `README.md` (setup)
2. Read: `README_QUICK_START.md` (overview)
3. Read: `README_LEGACY.md` sections on "Installation" & "Project Overview"

### Day 2: Level 1 - Simple Pendulum
1. Read: `pendulum_ode/README_DETAILED.md`
2. Review: `pendulum_ode/pendulum_ode.py` (understand equation)
3. Run: `pendulum_ode/pendulum_solver.py`
4. Analyze: `pendulum_ode/pendulum_solver_output.npz`
5. Review: outputs from `pendulum_ode/plot_results_pendulum.py`
6. Modify: `pendulum_ode/conf/config.yaml` (experiment)

### Day 3: Level 2 - Damped Pendulum
1. Read: `damped_pendulum/README_DETAILED.md`
2. Review: `damped_pendulum/damped_pendulum.py`
3. Run: `damped_pendulum/damped_pendulum_solver.py`
4. Study: differences from Level 1
5. Experiment: change `zeta` value

### Day 4: Level 3 - Inverse Problem
1. Read: `damped_pendulum_inverse/README_DETAILED.md`
2. Phase 1: Run `damped_pendulum_inverse/damped_pendulum_encoder_solver.py`
3. Analyze: `plot_results_encoder.py` output
4. Phase 2: Run `damped_pendulum_inverse/damped_pendulum_inverse_solver.py`
5. Final: Review `plot_results_inverse.py` results

---

## 📖 How to Find Information

### "I want to understand the physics"
→ [README_LEGACY.md](README_LEGACY.md) - Search for "Physics Background" section

### "I want to run System X"
→ [System/README_DETAILED.md](damped_pendulum/README_DETAILED.md) - How to Run section

### "System X is giving wrong results"
→ [System/README_DETAILED.md](damped_pendulum/README_DETAILED.md) - Troubleshooting section

### "I want to change hyperparameters"
→ [README_LEGACY.md](README_LEGACY.md) - Configuration Files section

### "I don't know where to start"
→ [README_QUICK_START.md](README_QUICK_START.md) - This file!

### "I want to understand the code"
→ System files ([*_ode.py](pendulum_ode/pendulum_ode.py), [*_solver.py](pendulum_ode/pendulum_solver.py))

### "I need to debug training"
→ [README_LEGACY.md](README_LEGACY.md) - Debugging Workflow & Implementation Notes

---

## ✅ Reading Paths by Goal

### Goal: Understand PINNs
1. [README_QUICK_START.md](README_QUICK_START.md) - Overview
2. [pendulum_ode/README_DETAILED.md](pendulum_ode/README_DETAILED.md) - Basics
3. [README_LEGACY.md](README_LEGACY.md) - Deep dive
4. Study: [pendulum_ode/pendulum_solver.py](pendulum_ode/pendulum_solver.py) - Code

### Goal: Get Results Quickly
1. [README_QUICK_START.md](README_QUICK_START.md) - Choose system
2. System's [README_DETAILED.md](#) - Run section
3. Execute the code
4. View PNG outputs

### Goal: Research Publication
1. [README_LEGACY.md](README_LEGACY.md) - Complete reference
2. Each system's [README_DETAILED.md](#) - Details
3. Source code for exact hyperparameters
4. NPZ files for data export

### Goal: Teach Others
1. [README_QUICK_START.md](README_QUICK_START.md) - Explain navigation
2. [pendulum_ode/README_DETAILED.md](pendulum_ode/README_DETAILED.md) - First example
3. Run together, examine outputs
4. Refer to [README_LEGACY.md](README_LEGACY.md) for deeper questions

---

## 📞 Quick Reference

| Need Help With | Go To | Section |
|---|---|---|
| Installation | [README.md](README.md) | Whole file |
| Overview | [README_QUICK_START.md](README_QUICK_START.md) | Project Overview |
| Physics | [README_LEGACY.md](README_LEGACY.md) | Physics Background |
| Level 1 System | [pendulum_ode/README_DETAILED.md](pendulum_ode/README_DETAILED.md) | Whole file |
| Level 2 System | [damped_pendulum/README_DETAILED.md](damped_pendulum/README_DETAILED.md) | Whole file |
| Level 3 System | [damped_pendulum_inverse/README_DETAILED.md](damped_pendulum_inverse/README_DETAILED.md) | Whole file |
| Running code | System's [README_DETAILED.md](#) | How to Run |
| Troubleshooting | System's [README_DETAILED.md](#) | Troubleshooting |
| Configuration | [README_LEGACY.md](README_LEGACY.md) | Configuration Files |
| Hyperparameters | [README_LEGACY.md](README_LEGACY.md) | Parameter Tuning Guide |
| Algorithm details | [README_LEGACY.md](README_LEGACY.md) | Detailed Algorithm |

---

**Document Created:** January 28, 2026  
**Last Updated:** January 28, 2026  
**Completeness:** 100% (all files documented)

For quick navigation, start with [README_QUICK_START.md](README_QUICK_START.md)! 🚀
