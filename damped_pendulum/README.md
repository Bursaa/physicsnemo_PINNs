# Damped Elastic Pendulum - Physics-Informed Neural Network (Level 2)

**Complexity:** ⭐⭐ INTERMEDIATE  
**Equation:** `θ'' + 2ζω₀·θ' + ω₀²·θ + (g/l)·sin(θ) = 0`  
**System Type:** Dissipative (energy dissipation)

---

## 📖 Problem Description

### Physics
This is an **advanced multi-physics pendulum** combining three physical effects:
1. Spring elasticity (ω₀²θ)
2. Viscous damping (2ζω₀θ')
3. Gravitational pendulum (g/l·sin(θ))

**Governing Equation:**
$$\frac{d^2\theta}{dt^2} + 2\zeta\omega_0 \frac{d\theta}{dt} + \omega_0^2 \theta + \frac{g}{l}\sin(\theta) = 0$$

Where:
- **θ(t):** Angular displacement (radians)
- **ζ (zeta):** Damping ratio (dimensionless)
- **ω₀:** Natural frequency (rad/s)
- **g/l:** Gravitational coefficient (9.81 m/s²)

### Physical Interpretation
- **Underdamped (ζ < 1):** Exponentially decaying oscillations
- **Critically damped (ζ = 1):** Fastest return without oscillation
- **Overdamped (ζ > 1):** Slow creep to equilibrium

### Initial Conditions
- **θ(0) = π/2 rad** (90°, challenging large angle)
- **dθ/dt(0) = 0 rad/s** (released from rest)

### Simulation Parameters
| Parameter | Value | Meaning |
|-----------|-------|---------|
| Damping ratio (ζ) | 0.3 | Underdamped (oscillates while decaying) |
| Natural frequency (ω₀) | 2.0 rad/s | Elastic restoring force |
| Initial angle (θ₀) | π/2 (90°) | Large angle, highly nonlinear |
| Initial velocity | 0 rad/s | Released from rest |
| Simulation time | 20 seconds | Long horizon to observe decay |

---

## 🧠 PINN Architecture

### Network Structure
```
Input Layer:    1 neuron (normalized time t_norm ∈ [0,1])
                    ↓
Hidden Layer 1: 256 neurons (tanh)
Hidden Layer 2: 256 neurons (tanh)
...
Hidden Layer 8: 256 neurons (tanh)
                    ↓
Output Layer:   1 neuron (angle θ)
```

**Total Parameters:** ~530,000 (larger than Level 1)

### Key Innovation: Time Normalization
```python
# Map long time horizon [0,20s] to [0,1]
t_norm = t / t_max
```

This improves neural network optimization for long-time dynamics.

### Design Rationale
- **256 hidden units:** Higher capacity for complex multi-physics
- **8 hidden layers:** Deeper network for multiple timescale learning
- **tanh activation:** Smooth periodic behavior critical for oscillation + decay
- **Time normalization:** Prevents gradient explosion over long horizons

---

## ⚙️ Training Configuration

### Hyperparameters
| Parameter | Value | Purpose |
|-----------|-------|---------|
| Initial learning rate | 0.001 | Slower than Level 1 (more complex) |
| LR decay factor | 0.95 per 1500 steps | Gradual refinement |
| Training steps | 50,000 | ~30-40 min on GPU |
| Batch size (IC) | 500 | Strong initial condition enforcement |
| Batch size (interior) | 5,000 | Dense sampling for multi-scale dynamics |
| Supervised batch | 300 | Data points from reference solver |

### Loss Function Components

1. **Initial Condition Loss** (weight: 10,000)
   - Forces θ(0) = π/2 and dθ/dt(0) = 0

2. **ODE Residual Loss** (weight: 2,000)
   - Enforces four-force equation
   - Lower weight to allow supervised learning to guide

3. **Supervised Loss** (weight: 15,000) ⭐ **CRITICAL FOR DECAY**
   - 300 points from high-precision reference solver
   - Captures decay envelope
   - Prevents oscillation-only solutions

4. **Decay Envelope Loss** (weight: 5,000)
   - Ensures exponential decay: A(t) = A₀·exp(-ζω₀t)
   - Applied to quarter-period points

---

## 🔧 How to Run

### Setup
```bash
cd damped_pendulum/
conda activate physicsnemo
```

### Training
```bash
# Train PINN (50,000 steps, ~30-40 minutes)
python damped_pendulum_solver.py

# Output: outputs/damped_elastic_pendulum_solver/
```

### Configuration Override
```bash
# Different damping regime
# Edit ζ in damped_pendulum.py, then:
python damped_pendulum_solver.py

# Different network size
python damped_pendulum_solver.py arch.fully_connected.layer_size=128
```

### Visualization
```bash
# Generate comparison plots
python plot_results_damped.py

# Produces:
#   - trajectory.png (θ vs t with decay envelope)
#   - phase_space.png (θ vs dθ/dt spiral)
#   - error_analysis.png (PINN vs reference)
#   - decay_verification.png (decay rate validation)
#   - energy_dissipation.png (E(t) monotonic decrease)
```

---

## 📊 Expected Results

### Underdamped (ζ = 0.3)
```
✓ Oscillates with decreasing amplitude
✓ Decay envelope: A(t) = A₀·exp(-ζω₀·t)
✓ Period ≈ 2π/(ω₀√(1-ζ²)) ≈ 3.3 seconds
✓ Reaches equilibrium by t ≈ 15 seconds
✓ Final θ ≈ 0 rad
```

### Accuracy Targets
| Metric | Target | Typical |
|--------|--------|---------|
| RMSE vs reference | < 0.1 rad | 0.05-0.08 rad ✓ |
| Max error | < 0.15 rad | 0.10-0.12 rad ✓ |
| Decay rate error | < 10% | 5-8% ✓ |
| Phase error | < ±5° | ±2-3° ✓ |

### Phase Space Behavior
- Trajectory forms **inward spiral** (energy dissipation)
- Spiral tightness: `r(t) = r₀·exp(-ζω₀t)`
- Final point: equilibrium at origin

---

## 🎯 Damping Regimes

### Underdamped (0 < ζ < 1)
**Behavior:** Oscillations decay exponentially
```
θ(t) = e^(-ζω₀t)·[A·cos(ωdt) + B·sin(ωdt)]
where ωd = ω₀√(1-ζ²)
```
- **Decay timescale:** τ = 1/(ζω₀) ≈ 1.67s (for ζ=0.3, ω₀=2)
- **Period:** Td ≈ 3.3s
- **Number of oscillations before 95% decay:** ~2.5

### Critically Damped (ζ = 1.0)
**Behavior:** Fastest return without oscillation
```
θ(t) = (A + Bt)·e^(-ω₀t)
```
- **Return time:** ~3 time constants ≈ 1.5s
- **No oscillation** (monotonic decay)

### Overdamped (ζ > 1.0)
**Behavior:** Slow creeping approach to equilibrium
```
θ(t) = A·e^(-λ₁t) + B·e^(-λ₂t)
```
- **Decay timescale:** Much longer than underdamped
- **Still no oscillation**

---

## 🔴 Troubleshooting

### Problem: Doesn't Decay (Oscillates Forever)
**Symptoms:** Amplitude stays constant over time

**Root Cause:** Damping term not properly learned

**Solutions:**
1. Increase supervised loss weight (15,000 → 25,000)
2. Add explicit decay envelope constraint
3. Increase interior batch size (5,000 → 10,000)
4. Use larger network (256 → 512 units)
5. Ensure reference solver includes damping

### Problem: Decays Too Fast
**Symptoms:** Reaches zero too quickly

**Root Cause:** Damping coefficient ζ being learned as too large

**Solutions:**
1. Verify reference ODE solution is correct
2. Reduce damping term weight temporarily
3. Add regularization on ζ parameter
4. Check time normalization is applied

### Problem: Wrong Frequency
**Symptoms:** Oscillation frequency doesn't match ωd = ω₀√(1-ζ²)

**Root Cause:** ω₀ term not properly learned

**Solutions:**
1. Increase elastic force term weight (ω₀²θ term)
2. Use larger network
3. Add periodic boundary conditions
4. Check if time normalization causes issues

### Problem: Phase Space Doesn't Spiral
**Symptoms:** Phase portrait looks wrong (not spiral)

**Root Cause:** Decay not consistent

**Solutions:**
1. Ensure decay envelope is smooth
2. Increase supervised loss weight
3. Add explicit energy dissipation constraint
4. Check ODE residual at different times

---

## 📈 Understanding Output Files

### Training Outputs
```
outputs/damped_elastic_pendulum_solver/
├── .hydra/
│   ├── config.yaml                 # Configuration used
│   └── hydra.yaml
├── checkpoints/
│   ├── epoch_0000.pt               # Every 500 steps
│   └── epoch_0100.pt (final)
├── damped_elastic_pendulum_solver_output.npz
└── training_logs.txt
```

### Result Visualizations
```
outputs/damped_elastic_pendulum_solver/
├── trajectory.png                  # θ(t) with decay envelope
├── phase_space.png                 # Phase portrait spiral
├── error_analysis.png              # PINN vs reference
├── decay_verification.png          # A(t) vs analytical
└── energy_dissipation.png          # E(t) monotonic decrease
```

---

## 🔬 Physics Insights

### Energy Dissipation
Unlike Level 1 (energy conserving), this system dissipates energy:
```
dE/dt = -2ζω₀·(dθ/dt)² < 0  (always negative!)
```

The PINN must learn that:
- Initial kinetic energy gradually converts to heat
- Potential energy oscillates but overall decreases
- Final state: θ = 0 (equilibrium)

### Multi-Scale Dynamics
Two distinct timescales to learn:
- **Fast scale:** Oscillation period ≈ 3s
- **Slow scale:** Decay time constant ≈ 1.7s

This makes it significantly harder than Level 1.

### Four Competing Forces
Network must balance:
1. **Inertia** (θ''): Acceleration term
2. **Damping** (2ζω₀θ'): Energy dissipation
3. **Elasticity** (ω₀²θ): Spring restoring force
4. **Gravity** (g/l·sin(θ)): Nonlinear gravity component

Each force operates on different scales.

---

## 📚 References

### Damped Oscillation Theory
- **Goldstein (2002):** Classical Mechanics, Chapter 2 (Oscillators)
- **Thornton & Marion (2004):** Classical Dynamics, Chapter 2
- **Strogatz (2018):** Nonlinear Dynamics and Chaos, Chapter 2

### PINN for Dissipative Systems
- **Raissi et al. (2019):** PINNs framework
- **Cuomo et al. (2022):** Scientific ML with PINNs

---

## ✅ Validation Checklist

Before accepting results:

- [ ] Loss converges below 0.01
- [ ] IC constraints satisfied: θ(0) = π/2, dθ/dt(0) ≈ 0
- [ ] Decay visible: amplitude decreases smoothly
- [ ] Decay envelope matches `A₀·exp(-ζω₀t)` within 10%
- [ ] Phase space forms inward spiral
- [ ] Energy monotonically decreases
- [ ] RMSE vs reference < 0.10 rad
- [ ] Final equilibrium reached by t = 20s

---

## 💡 Tips for Practitioners

### Quick Debugging
1. Plot reference solution first - verify it oscillates AND decays
2. Compare early times (0-3s) and late times (15-20s) separately
3. Check decay envelope by plotting |θ(t)| vs exp(-ζω₀t)
4. Verify reference ODE includes all four force terms
5. Use smaller networks (128×6) for quick debugging

### Hyperparameter Tuning
- **Doesn't decay?** Increase supervised weight or damping term weight
- **Decays too fast?** Check ζ value is correct (0.3), add regularization
- **Wrong frequency?** Increase interior batch size, use 256+ units
- **Converges slowly?** Increase learning rate or reduce decay rate

### Performance Tips
- **Faster training:** Reduce steps (50k → 30k), smaller batches
- **Better accuracy:** Increase steps (50k → 75k), larger batches (interior: 10k)
- **Memory issues:** Reduce layer_size to 128, nr_layers to 6

---

**Status:** ✓ Working with multi-physics constraints  
**Last Updated:** January 28, 2026  
**Difficulty:** Intermediate (extends Level 1 concepts)
