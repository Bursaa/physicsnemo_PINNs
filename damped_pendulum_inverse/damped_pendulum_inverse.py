# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Damped Pendulum ODE for Inverse Problem
Equation: d²θ/dt² + ζ·dθ/dt + (g/l)·sin(θ) = 0

In the inverse problem, ζ (zeta) is a learnable parameter that we want to infer
from observational data.
"""

from sympy import Symbol, Function, sin, S, diff
from physicsnemo.sym.eq.pde import PDE


class DampedPendulumInverse(PDE):
    """
    Damped Pendulum equation for inverse problem:
    d²θ/dt² + ζ·dθ/dt + (g/l)·sin(θ) = 0

    Here ζ (zeta) is a Symbol that will be predicted by a separate network,
    allowing us to infer the damping coefficient from data.

    Parameters
    ==========
    g : float
        Gravitational acceleration (default: 9.81 m/s²)
    l : float
        Pendulum length (default: 1.0 m)
    """

    name = "DampedPendulumInverse"

    def __init__(self, g=9.81, l=1.0):
        self.g = g
        self.l = l

        # time variable
        t = Symbol("t")
        input_variables = {"t": t}

        # angle function
        theta = Function("theta")(*input_variables)
        # zeta is now a Symbol - it will be predicted by a network
        zeta = Symbol("zeta")

        # convert parameters to symbolic constants
        g_sym = S(g)
        l_sym = S(l)

        # set equation: θ'' + ζ·θ' + (g/l)·sin(θ) = 0
        # Use theta.diff() method directly to ensure proper Expr type
        self.equations = {}
        self.equations["damped_pendulum_inverse"] = (
            theta.diff(t, 2)
            + zeta * theta.diff(t)
            + (g_sym / l_sym) * sin(theta)
        )
