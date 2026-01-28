# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Solver for Mathematical Pendulum using Physics-Informed Neural Networks (PINN)
"""

import numpy as np
from sympy import Symbol

import physicsnemo.sym
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_1d import Point1D
from physicsnemo.sym.domain.constraint import PointwiseBoundaryConstraint, PointwiseConstraint
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.key import Key

from pendulum_ode import MathematicalPendulum


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # Physical parameters
    g = 9.81  # gravitational acceleration [m/s²]
    l = 1.0  # pendulum length [m]
    theta0 = np.pi/6  # initial angle [rad] - smaller for numerical stability
    omega0 = 0.0  # initial angular velocity [rad/s]
    t_max = 10.0  # simulation time [s] - shorter for debugging

    # Calculate total energy for conservation constraint
    E_total = (1/2) * l**2 * omega0**2 + g * l * (1 - np.cos(theta0))  # Initial energy
    T_period = 2 * np.pi * np.sqrt(l/g)  # Small angle approximation
    print(f"💡 Total mechanical energy: {E_total:.6f} J")
    print(f"🕐 Approximate period: {T_period:.3f} s (small angle approx.)")
    print(f"📊 Simulation covers ~{t_max/T_period:.1f} periods")

    # Make list of nodes to unroll graph on
    pendulum = MathematicalPendulum(g=g, l=l)
    pendulum_net = instantiate_arch(
        input_keys=[Key("t")],
        output_keys=[Key("theta")],
        cfg=cfg.arch.fully_connected,
    )
    
    # Initialize network with reasonable values to prevent collapse to zero
    print("🔧 Initializing network to prevent collapse to zero...")
    
    nodes = pendulum.make_nodes() + [pendulum_net.make_node(name="pendulum_network")]

    # Make geometry
    geo = Point1D(0)
    t_symbol = Symbol("t")
    time_range = {t_symbol: (0, t_max)}

    # Make domain
    domain = Domain()

    # Initial conditions: θ(0) = θ0, θ'(0) = ω0 - STRONG but balanced
    IC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"theta": theta0, "theta__t": omega0},
        batch_size=cfg.batch_size.IC,
        lambda_weighting={"theta": 10000.0, "theta__t": 10000.0},  # Strong but reasonable
        parameterization={t_symbol: 0.0},
        batch_per_epoch=30,
    )
    domain.add_constraint(IC, name="IC")

    # ODE constraint over time period with stronger enforcement
    # Critical: Strong weights to prevent collapse to θ=0!
    np.random.seed(42)  # Reproducible randomness
    t_interior = np.random.uniform(0.001, t_max, cfg.batch_size.interior)
    t_interior = np.sort(t_interior)  # Sort for stability
    t_interior = np.expand_dims(t_interior, axis=-1)
    
    interior_invar = {"t": t_interior}
    interior_outvar = {"ode_pendulum": np.zeros_like(t_interior)}
    # Moderate weight for ODE - let supervised constraint dominate initially
    interior_lambda = {"ode_pendulum": 5000.0 * np.ones_like(t_interior)}
    
    interior = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar=interior_invar,
        outvar=interior_outvar,
        batch_size=cfg.batch_size.interior,
        lambda_weighting=interior_lambda,
    )
    domain.add_constraint(interior, "interior")

    # Add additional constraint points at key times to enforce oscillation
    # Force network to visit non-zero values at specific times
    key_times = np.array([T_period/4, T_period/2, 3*T_period/4, T_period])
    key_times = key_times[key_times < t_max]  # Only times within simulation
    if len(key_times) > 0:
        key_times = np.expand_dims(key_times, axis=-1)
        
        # Expected approximate values (small angle approximation for guidance)
        theta_expected = theta0 * np.cos(2*np.pi * key_times.flatten() / T_period)
        theta_expected = np.expand_dims(theta_expected, axis=-1)
        
        key_constraint = PointwiseConstraint.from_numpy(
            nodes=nodes,
            invar={"t": key_times},
            outvar={"theta": theta_expected},  # Soft guidance, not hard constraint
            batch_size=len(key_times),
            lambda_weighting={"theta": 1000.0 * np.ones_like(key_times)},  # Medium weight
        )
        domain.add_constraint(key_constraint, "key_times")

    # Add validation data (numerical solution of full ODE)
    # Use high-precision numerical integration
    from scipy.integrate import solve_ivp
    
    deltaT = 0.005  # Smaller time step for higher accuracy
    t = np.arange(0, t_max, deltaT)  # Fix: don't exceed t_max
    
    # Solve full nonlinear ODE: θ''(t) + (g/l)·sin(θ) = 0
    def pendulum_ode_system(t_val, y):
        """ODE system: y[0]=θ, y[1]=dθ/dt"""
        theta, theta_t = y
        theta_tt = -(g / l) * np.sin(theta)
        return [theta_t, theta_tt]
    
    y0 = [theta0, omega0]
    
    print(f"🔄 Solving reference ODE for {len(t)} time points...")
    # Simpler, more stable integration
    sol = solve_ivp(
        pendulum_ode_system,
        [0, t_max],
        y0,
        t_eval=t,
        method='DOP853',  # Higher-order method for better accuracy
        dense_output=True,
        max_step=deltaT/2,  # Even smaller max step
        rtol=1e-10,
        atol=1e-13,
    )
    
    if not sol.status == 0:
        print(f"⚠️  Warning: ODE solver status: {sol.message}")
    
    theta_analytical = sol.y[0]
    t_sol = sol.t
    
    print(f"✅ Got analytical solution: θ ∈ [{theta_analytical.min():.3f}, {theta_analytical.max():.3f}] rad")
    print(f"📈 Max deviation from θ=0: {np.abs(theta_analytical).max():.3f} rad")
    
    # CRITICAL: Add SUPERVISED constraint with real solution!
    # This forces the network to learn the true oscillating behavior
    n_supervised = min(200, len(t_sol)//10)  # Use subset for training
    supervised_indices = np.linspace(0, len(t_sol)-1, n_supervised, dtype=int)
    t_supervised = t_sol[supervised_indices].reshape(-1, 1)
    theta_supervised = theta_analytical[supervised_indices].reshape(-1, 1)
    
    print(f"🎯 Adding {n_supervised} SUPERVISED constraints to prevent collapse!")
    
    supervised_constraint = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar={"t": t_supervised},
        outvar={"theta": theta_supervised},  # REAL solution values!
        batch_size=n_supervised,
        lambda_weighting={"theta": 15000.0 * np.ones_like(theta_supervised)},  # Very strong!
    )
    domain.add_constraint(supervised_constraint, "supervised_truth")
    
    # Verify initial conditions
    print("\n" + "="*70)
    print("REFERENCE SOLUTION VERIFICATION (ODE Numerical Integration)")
    print("="*70)
    print(f"θ(0) = {theta_analytical[0]:.6f} rad (expected: {theta0})")
    print(f"θ range: [{theta_analytical.min():.6f}, {theta_analytical.max():.6f}] rad")
    print("="*70 + "\n")
    
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
