# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Solver for Damped Elastic Pendulum using Physics-Informed Neural Networks (PINN)
"""

import numpy as np
from sympy import Symbol, Number, Function

import physicsnemo.sym
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_1d import Point1D
from physicsnemo.sym.domain.constraint import PointwiseBoundaryConstraint, PointwiseConstraint
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node

from damped_elastic_pendulum import DampedElasticPendulum


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # Physical parameters
    zeta = 0.3  # damping ratio (underdamped oscillation)
    g = 9.81  # gravitational acceleration [m/s²]
    l = 1.0  # pendulum length [m]
    theta0 = np.pi/2  # initial angle [rad]
    omega_0 = 0.0  # initial angular velocity [rad/s]
    t_max = 20.0  # simulation time [s]

    # Time normalization: map t ∈ [0, t_max] to t_norm ∈ [0, 1]
    # This helps the neural network learn better by keeping inputs in a bounded range
    t_sym = Symbol("t")
    t_norm = t_sym / t_max  # Normalized time
    
    # Create normalization node: t -> t_norm
    time_norm_node = Node.from_sympy(t_norm, "t_norm")
    
    # Make list of nodes to unroll graph on
    damped_pendulum = DampedElasticPendulum(zeta=zeta, g=g, l=l)
    damped_net = instantiate_arch(
        input_keys=[Key("t_norm")],  # Network takes normalized time
        output_keys=[Key("theta")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = [time_norm_node] + damped_pendulum.make_nodes() + [damped_net.make_node(name="damped_network")]

    # Make geometry
    geo = Point1D(0)
    t_symbol = Symbol("t")
    time_range = {t_symbol: (0, t_max)}

    # Make domain
    domain = Domain()

    # Initial conditions: θ(0) = θ0, θ'(0) = ω0
    # Critical: Use very high weights to enforce IC strongly
    IC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"theta": theta0, "theta__t": omega_0},
        batch_size=cfg.batch_size.IC,
        lambda_weighting={"theta": 100.0, "theta__t": 100.0},
        parameterization={t_symbol: 0.0},
        batch_per_epoch=2000,  # Increased significantly for better IC enforcement
    )
    domain.add_constraint(IC, name="IC")

    # ODE constraint over time period
    # Sample time points for interior constraint - start from small positive value
    # Use more points near t=0 where dynamics are strongest (cosine spacing)
    n_interior = 50000  # Fixed number of interior points
    
    # Cosine spacing: more points near the beginning
    u = np.linspace(0, 1, n_interior)
    t_interior = t_max * (1 - np.cos(np.pi * u / 2))  # Cosine spacing from 0 to t_max
    t_interior[0] = 0.001  # Avoid exact zero (handled by IC)
    t_interior = np.expand_dims(t_interior, axis=-1)
    
    interior_invar = {"t": t_interior}
    interior_outvar = {"damped_elastic_pendulum": np.zeros_like(t_interior)}
    interior_lambda = {"damped_elastic_pendulum": np.ones_like(t_interior)}
    
    interior = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar=interior_invar,
        outvar=interior_outvar,
        batch_size=cfg.batch_size.interior,
        lambda_weighting=interior_lambda,
    )
    domain.add_constraint(interior, "interior")

    # Add validation data (numerical solution of full ODE)
    # Use high-precision numerical integration
    deltaT = 0.005  # Smaller time step for higher accuracy
    t = np.arange(0, t_max + deltaT, deltaT)
    
    # Solve full nonlinear ODE: θ'' + 2ζω₀·θ' + ω₀²·θ + (g/l)·sin(θ) = 0
    def damped_ode(t_val, y):
        """
        ODE system for damped elastic pendulum
        y[0] = θ (angle)
        y[1] = dθ/dt (angular velocity)
        """
        theta, theta_t = y
        # d²θ/dt² = -ζ·dθ/dt - (g/l)·sin(θ)
        theta_tt = -(zeta * theta_t + (g / l) * np.sin(theta))
        return [theta_t, theta_tt]
    
    y0 = [theta0, omega_0]
    
    # Solve using RK45 (higher order, adaptive step) for reference solution
    from scipy.integrate import solve_ivp
    
    # High-precision solution
    sol = solve_ivp(
        damped_ode, 
        [0, t_max], 
        y0, 
        t_eval=t,
        method='DOP853',
        dense_output=True,
        max_step=0.001,  # Maximum step size for accuracy
        rtol=1e-9,  # Relative tolerance
        atol=1e-12,  # Absolute tolerance
    )
    
    if not sol.status == 0:
        print(f"⚠️  Warning: ODE solver warning/error: {sol.message}")
    
    # Extract solution
    theta_analytical = sol.y[0]  # θ(t)
    theta_dot_analytical = sol.y[1]  # dθ/dt(t)
    t_sol = sol.t

# Verify physical correctness of reference solution
    print("\n" + "="*70)
    print("REFERENCE SOLUTION VERIFICATION (ODE Numerical Integration)")
    print("="*70)
    print(f"Time span: {t_sol[0]:.3f} to {t_sol[-1]:.3f} seconds")
    print(f"Number of time points: {len(t_sol)}")
    print(f"θ(0) = {theta_analytical[0]:.6f} rad (should be {theta0})")
    print(f"θ'(0) = {theta_dot_analytical[0]:.6f} rad/s (should be {omega_0})")
    print(f"θ range: [{theta_analytical.min():.6f}, {theta_analytical.max():.6f}] rad")
    
    # Check damping (amplitude should decrease)
    max_early = np.abs(theta_analytical[:len(theta_analytical)//2]).max()
    max_late = np.abs(theta_analytical[len(theta_analytical)//2:]).max()
    print(f"Amplitude decay: {max_early:.6f} → {max_late:.6f} (ratio: {max_late/max_early:.4f})")
    print(f"Expected: ratio should be < 1.0 for damped system")
    
    if max_late / max_early >= 1.0:
        print("⚠️  WARNING: Amplitude not decaying! Check damping parameters!")
    else:
        print("✓ Damping behavior verified")
    print("="*70 + "\n")
    
    # Reshape for validation
    t_val = np.expand_dims(t_sol, axis=-1)
    theta_val = np.expand_dims(theta_analytical, axis=-1)

    invar_numpy = {"t": t_val}
    outvar_numpy = {"theta": theta_val}

    validator = PointwiseValidator(
        nodes=nodes, invar=invar_numpy, true_outvar=outvar_numpy, batch_size=1024
    )
    domain.add_validator(validator)

    # Make solver
    slv = Solver(cfg, domain)

    # Start solver
    slv.solve()


if __name__ == "__main__":
    run()
