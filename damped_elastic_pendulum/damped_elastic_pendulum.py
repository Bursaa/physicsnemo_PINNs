# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Damped Elastic Pendulum ODE
Equation: d²θ/dt² + 2ζω₀·dθ/dt + ω₀²·θ + (g/l)·sin(θ) = 0

Physical interpretation:
- First term: inertia (mass acceleration)
- Second term: damping (friction/air resistance)
- Third term: elastic restoring force (spring)
- Fourth term: gravitational restoring force (pendulum)

Parameters:
- ζ (zeta): damping ratio (0 = undamped, 0 < ζ < 1 = underdamped, ζ = 1 = critically damped, ζ > 1 = overdamped)
- ω₀ (omega0): natural frequency of the system
- g/l: gravitational restoring coefficient
"""

from sympy import Symbol, Function, Number, sin
from physicsnemo.sym.eq.pde import PDE


class DampedElasticPendulum(PDE):
    """
    Damped Elastic Pendulum equation:
    d²θ/dt² + 2ζω₀·dθ/dt + ω₀²·θ + (g/l)·sin(θ) = 0

    Parameters
    ==========
    zeta : float
        Damping ratio (dimensionless)
        - 0 < zeta < 1: underdamped (oscillatory with decay)
        - zeta = 1: critically damped
        - zeta > 1: overdamped (no oscillation)
    omega0 : float
        Natural frequency of oscillation (rad/s)
    g : float
        Gravitational acceleration (default: 9.81 m/s²)
    l : float
        Pendulum length (default: 1.0 m)
    """

    name = "DampedElasticPendulum"

    def __init__(self, zeta=0.3, g=9.81, l=1.0):
        self.zeta = zeta
        self.g = g
        self.l = l

        # time variable
        t = Symbol("t")
        input_variables = {"t": t}

        # angle function
        theta = Function("theta")(*input_variables)

        # convert parameters to symbolic constants
        if type(zeta) in [float, int]:
            zeta = Number(zeta)
        if type(g) in [float, int]:
            g = Number(g)
        if type(l) in [float, int]:
            l = Number(l)

        # set equation: θ'' + ζ·θ' + (g/l)·sin(θ) = 0
        self.equations = {}
        self.equations["damped_elastic_pendulum"] = (
            theta.diff(t, 2) 
            + zeta * theta.diff(t)
            + (g / l) * sin(theta)
        )
