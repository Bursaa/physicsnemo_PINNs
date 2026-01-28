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
Mathematical Pendulum ODE
Equation: d²θ/dt² + (g/l)·sin(θ) = 0
"""

from sympy import Symbol, Function, Number, sin
from physicsnemo.sym.eq.pde import PDE


class MathematicalPendulum(PDE):
    """
    Mathematical pendulum equation: d²θ/dt² + (g/l)·sin(θ) = 0

    Parameters
    ==========
    g : float
        Gravitational acceleration (default: 9.81 m/s²)
    l : float
        Pendulum length (default: 1.0 m)
    """

    name = "MathematicalPendulum"

    def __init__(self, g=9.81, l=1.0):
        self.g = g
        self.l = l

        # time variable
        t = Symbol("t")
        input_variables = {"t": t}

        # angle function
        theta = Function("theta")(*input_variables)

        # convert parameters to symbolic constants
        if type(g) is str:
            g = Function(g)(*input_variables)
        elif type(g) in [float, int]:
            g = Number(g)

        if type(l) is str:
            l = Function(l)(*input_variables)
        elif type(l) in [float, int]:
            l = Number(l)

        # set equation: θ'' + (g/l)·sin(θ) = 0
        self.equations = {}
        self.equations["ode_pendulum"] = theta.diff(t, 2) + (g / l) * sin(theta)
