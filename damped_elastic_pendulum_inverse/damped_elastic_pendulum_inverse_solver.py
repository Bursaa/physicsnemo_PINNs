# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Inverse PINN Solver for Damped Pendulum

This solver infers the unknown parameters (zeta, theta0, omega0) from observational data.
The approach:
1. A neural network predicts theta(t) given time t
2. A parameter network outputs the unknown parameters (zeta, theta0, omega0)
3. The physics constraint (ODE) is enforced
4. The data constraint matches observations
"""

import torch
import torch.nn as nn
import numpy as np
from sympy import Symbol

import physicsnemo.sym
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_1d import Point1D, Line1D
from physicsnemo.sym.domain.constraint import PointwiseBoundaryConstraint, PointwiseConstraint
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.models.arch import Arch

from damped_elastic_pendulum_inverse import DampedPendulumInverse


class ParameterNetArch(Arch):
    """
    A simple network that outputs learnable parameters.
    This network takes no input and outputs constant parameter values that are learned during training.
    
    For the inverse pendulum problem, it outputs:
    - zeta: damping coefficient
    - theta0: initial angle (estimated)
    - omega0: initial angular velocity (estimated)
    """
    
    def __init__(
        self,
        output_keys: list,
        zeta_init: float = 0.5,
        theta0_init: float = 1.0,
        omega0_init: float = 0.0,
    ):
        super().__init__(
            input_keys=[],
            output_keys=output_keys,
            periodicity=None,
        )
        
        # Learnable parameters with reasonable initial guesses
        # Using raw values and applying transformations to ensure physical bounds
        self.zeta_raw = nn.Parameter(torch.tensor([self._inverse_softplus(zeta_init)], dtype=torch.float32))
        self.theta0_raw = nn.Parameter(torch.tensor([theta0_init], dtype=torch.float32))
        self.omega0_raw = nn.Parameter(torch.tensor([omega0_init], dtype=torch.float32))
        
    def _inverse_softplus(self, x):
        """Inverse of softplus for initialization"""
        return np.log(np.exp(x) - 1 + 1e-8)
    
    def forward(self, invar):
        # Get batch size from any input (or use 1 if no inputs)
        batch_size = 1
        for key in invar:
            batch_size = invar[key].shape[0]
            device = invar[key].device
            break
        else:
            device = self.zeta_raw.device
        
        # Apply softplus to zeta to ensure it's positive (damping must be >= 0)
        zeta = torch.nn.functional.softplus(self.zeta_raw)
        
        # theta0 and omega0 can be any real value
        theta0 = self.theta0_raw
        omega0 = self.omega0_raw
        
        # Expand to batch size
        return {
            "zeta": zeta.expand(batch_size, 1).to(device),
            "theta0": theta0.expand(batch_size, 1).to(device),
            "omega0": omega0.expand(batch_size, 1).to(device),
        }


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # ==========================================================================
    # TRUE PARAMETERS (we want to infer these from data)
    # ==========================================================================
    TRUE_ZETA = 0.3      # True damping coefficient
    TRUE_THETA0 = np.pi/2  # True initial angle [rad]
    TRUE_OMEGA0 = 0.0    # True initial angular velocity [rad/s]
    
    # Physical constants (known)
    g = 9.81  # gravitational acceleration [m/s²]
    l = 1.0   # pendulum length [m]
    t_max = 20.0  # simulation time [s]
    
    # Initial guesses for parameters (intentionally wrong)
    INIT_ZETA = 0.5      # Initial guess for zeta
    INIT_THETA0 = 1.0    # Initial guess for theta0
    INIT_OMEGA0 = 0.5    # Initial guess for omega0

    print("\n" + "="*70)
    print("INVERSE PINN: PARAMETER ESTIMATION FOR DAMPED PENDULUM")
    print("="*70)
    print(f"\nTrue parameters (to be inferred):")
    print(f"  ζ (zeta)  = {TRUE_ZETA}")
    print(f"  θ₀        = {TRUE_THETA0:.6f} rad ({np.degrees(TRUE_THETA0):.2f}°)")
    print(f"  ω₀        = {TRUE_OMEGA0} rad/s")
    print(f"\nInitial guesses:")
    print(f"  ζ (zeta)  = {INIT_ZETA}")
    print(f"  θ₀        = {INIT_THETA0:.6f} rad")
    print(f"  ω₀        = {INIT_OMEGA0} rad/s")
    print("="*70 + "\n")

    # ==========================================================================
    # GENERATE SYNTHETIC OBSERVATIONAL DATA
    # ==========================================================================
    from scipy.integrate import solve_ivp
    
    def generate_observation_data(zeta, theta0, omega0, t_eval, noise_level=0.0):
        """Generate synthetic observation data from the true ODE solution"""
        def damped_ode(t_val, y):
            theta, theta_t = y
            theta_tt = -(zeta * theta_t + (g / l) * np.sin(theta))
            return [theta_t, theta_tt]
        
        sol = solve_ivp(
            damped_ode, 
            [0, t_max], 
            [theta0, omega0], 
            t_eval=t_eval,
            method='DOP853',
            rtol=1e-10,
            atol=1e-12,
        )
        
        theta_data = sol.y[0]
        
        # Add noise if specified
        if noise_level > 0:
            theta_data = theta_data + noise_level * np.random.randn(len(theta_data))
        
        return theta_data
    
    # Generate observation data at sparse time points
    n_observations = cfg.custom.n_observations  # Number of observation points
    noise_level = cfg.custom.noise_level  # Noise level (0 = no noise)
    
    t_obs = np.linspace(0, t_max, n_observations)
    theta_obs = generate_observation_data(TRUE_ZETA, TRUE_THETA0, TRUE_OMEGA0, t_obs, noise_level)
    
    print(f"Generated {n_observations} observation points with noise level = {noise_level}")
    print(f"Observation time range: [{t_obs[0]:.2f}, {t_obs[-1]:.2f}] s")
    print(f"Observation θ range: [{theta_obs.min():.4f}, {theta_obs.max():.4f}] rad\n")

    # ==========================================================================
    # SETUP NEURAL NETWORKS
    # ==========================================================================
    
    # Time normalization node
    t_sym = Symbol("t")
    t_norm = t_sym / t_max
    time_norm_node = Node.from_sympy(t_norm, "t_norm")
    
    # Main network: predicts theta(t)
    theta_net = instantiate_arch(
        input_keys=[Key("t_norm")],
        output_keys=[Key("theta")],
        cfg=cfg.arch.fully_connected,
    )
    
    # Parameter network: outputs learnable parameters (zeta, theta0, omega0)
    param_net = ParameterNetArch(
        output_keys=[Key("zeta"), Key("theta0"), Key("omega0")],
        zeta_init=INIT_ZETA,
        theta0_init=INIT_THETA0,
        omega0_init=INIT_OMEGA0,
    )
    
    # Physics equation (with zeta as input from parameter network)
    physics_eq = DampedPendulumInverse(g=g, l=l)
    
    # Build node list
    nodes = (
        [time_norm_node] 
        + [param_net.make_node(name="parameter_network", optimize=True)]
        + physics_eq.make_nodes() 
        + [theta_net.make_node(name="theta_network")]
    )

    # ==========================================================================
    # SETUP DOMAIN AND CONSTRAINTS
    # ==========================================================================
    
    geo = Point1D(0)
    t_symbol = Symbol("t")
    
    domain = Domain()
    
    # --------------------------------------------------------------------------
    # 1. DATA CONSTRAINT: Match observations
    # --------------------------------------------------------------------------
    t_obs_arr = np.expand_dims(t_obs, axis=-1).astype(np.float32)
    theta_obs_arr = np.expand_dims(theta_obs, axis=-1).astype(np.float32)
    
    data_constraint = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar={"t": t_obs_arr},
        outvar={"theta": theta_obs_arr},
        batch_size=min(cfg.batch_size.data, n_observations),
        lambda_weighting={"theta": np.ones_like(theta_obs_arr) * cfg.custom.lambda_data},
    )
    domain.add_constraint(data_constraint, "data")
    
    # --------------------------------------------------------------------------
    # 2. INITIAL CONDITION CONSTRAINT: theta(0) = theta0, theta'(0) = omega0
    # Here theta0 and omega0 come from the parameter network
    # --------------------------------------------------------------------------
    # For IC, we need a special approach: enforce that at t=0, 
    # theta equals the learned theta0, and theta__t equals omega0
    
    # Create nodes that compute (theta - theta0) and (theta__t - omega0) at t=0
    # These should be zero
    theta_sym = Symbol("theta")
    theta0_sym = Symbol("theta0")
    theta_t_sym = Symbol("theta__t")
    omega0_sym = Symbol("omega0")
    
    ic_theta_node = Node.from_sympy(theta_sym - theta0_sym, "ic_theta")
    ic_omega_node = Node.from_sympy(theta_t_sym - omega0_sym, "ic_omega")
    
    ic_nodes = nodes + [ic_theta_node, ic_omega_node]
    
    IC = PointwiseBoundaryConstraint(
        nodes=ic_nodes,
        geometry=geo,
        outvar={"ic_theta": 0.0, "ic_omega": 0.0},
        batch_size=cfg.batch_size.IC,
        lambda_weighting={"ic_theta": cfg.custom.lambda_ic, "ic_omega": cfg.custom.lambda_ic},
        parameterization={t_symbol: 0.0},
        batch_per_epoch=1000,
    )
    domain.add_constraint(IC, name="IC")
    
    # --------------------------------------------------------------------------
    # 3. PHYSICS CONSTRAINT: ODE residual should be zero
    # --------------------------------------------------------------------------
    n_interior = 10000
    u = np.linspace(0, 1, n_interior)
    t_interior = t_max * (1 - np.cos(np.pi * u / 2))  # Cosine spacing
    t_interior[0] = 0.001
    t_interior = np.expand_dims(t_interior, axis=-1).astype(np.float32)
    
    interior = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar={"t": t_interior},
        outvar={"damped_pendulum_inverse": np.zeros_like(t_interior)},
        batch_size=cfg.batch_size.interior,
        lambda_weighting={"damped_pendulum_inverse": np.ones_like(t_interior) * cfg.custom.lambda_physics},
    )
    domain.add_constraint(interior, "interior")
    
    # --------------------------------------------------------------------------
    # 4. VALIDATOR: Compare against true solution
    # --------------------------------------------------------------------------
    t_val = np.linspace(0, t_max, 2001)
    theta_val = generate_observation_data(TRUE_ZETA, TRUE_THETA0, TRUE_OMEGA0, t_val, noise_level=0)
    
    t_val_arr = np.expand_dims(t_val, axis=-1).astype(np.float32)
    theta_val_arr = np.expand_dims(theta_val, axis=-1).astype(np.float32)
    
    validator = PointwiseValidator(
        nodes=nodes, 
        invar={"t": t_val_arr}, 
        true_outvar={"theta": theta_val_arr},
        batch_size=1024
    )
    domain.add_validator(validator)

    # ==========================================================================
    # SOLVE
    # ==========================================================================
    slv = Solver(cfg, domain)
    slv.solve()
    
    # ==========================================================================
    # PRINT FINAL RESULTS
    # ==========================================================================
    print("\n" + "="*70)
    print("FINAL PARAMETER ESTIMATION RESULTS")
    print("="*70)
    
    # Get final parameter values
    param_net.eval()
    with torch.no_grad():
        dummy_input = {"t": torch.zeros(1, 1)}
        params = param_net(dummy_input)
        final_zeta = params["zeta"].item()
        final_theta0 = params["theta0"].item()
        final_omega0 = params["omega0"].item()
    
    print(f"\n{'Parameter':<15} {'True Value':<15} {'Estimated':<15} {'Error':<15} {'Rel. Error':<15}")
    print("-"*75)
    
    zeta_err = abs(final_zeta - TRUE_ZETA)
    zeta_rel = 100 * zeta_err / TRUE_ZETA
    print(f"{'ζ (zeta)':<15} {TRUE_ZETA:<15.6f} {final_zeta:<15.6f} {zeta_err:<15.6f} {zeta_rel:<14.2f}%")
    
    theta0_err = abs(final_theta0 - TRUE_THETA0)
    theta0_rel = 100 * theta0_err / abs(TRUE_THETA0)
    print(f"{'θ₀':<15} {TRUE_THETA0:<15.6f} {final_theta0:<15.6f} {theta0_err:<15.6f} {theta0_rel:<14.2f}%")
    
    omega0_err = abs(final_omega0 - TRUE_OMEGA0)
    omega0_rel_str = f"{100*omega0_err:.2f}%" if TRUE_OMEGA0 != 0 else "N/A"
    print(f"{'ω₀':<15} {TRUE_OMEGA0:<15.6f} {final_omega0:<15.6f} {omega0_err:<15.6f} {omega0_rel_str:<15}")
    
    print("\n" + "="*70)
    
    # Save parameters to file for plotting
    np.savez(
        "outputs/damped_elastic_pendulum_inverse_solver/estimated_parameters.npz",
        true_zeta=TRUE_ZETA,
        true_theta0=TRUE_THETA0,
        true_omega0=TRUE_OMEGA0,
        estimated_zeta=final_zeta,
        estimated_theta0=final_theta0,
        estimated_omega0=final_omega0,
        init_zeta=INIT_ZETA,
        init_theta0=INIT_THETA0,
        init_omega0=INIT_OMEGA0,
    )
    print("Parameter estimates saved to outputs/damped_elastic_pendulum_inverse_solver/estimated_parameters.npz")


if __name__ == "__main__":
    run()
