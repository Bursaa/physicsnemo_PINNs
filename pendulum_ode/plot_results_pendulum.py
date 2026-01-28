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
Visualization script for Mathematical Pendulum PINN results
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Try to find the results directory
base_dir = "outputs/pendulum_solver/validators/"

if not os.path.exists(base_dir):
    print(f"Error: Directory {base_dir} not found!")
    print("Make sure to run pendulum_solver.py first.")
    exit(1)

# Load validation data
data_file = base_dir + "validator.npz"
if not os.path.exists(data_file):
    print(f"Error: File {data_file} not found!")
    print("Make sure the training completed successfully.")
    exit(1)

data = np.load(data_file, allow_pickle=True)
data = np.atleast_1d(data.f.arr_0)[0]

# Extract time and angles
t = data["t"].flatten()
theta_true = data["true_theta"].flatten()
theta_pred = data["pred_theta"].flatten()

# Calculate error
error = np.abs(theta_pred - theta_true)

# Create figure with subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Comparison of solutions
ax1 = axes[0]
ax1.plot(t, theta_true, "b-", linewidth=2, label="Analytical (small angle approx.)")
ax1.plot(t, theta_pred, "r--", linewidth=2, label="PINN prediction")
ax1.set_xlabel("Time [s]", fontsize=12)
ax1.set_ylabel("θ(t) [rad]", fontsize=12)
ax1.set_title("Mathematical Pendulum - PINN vs Analytical Solution", fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: Absolute error
ax2 = axes[1]
ax2.plot(t, error, "g-", linewidth=2)
ax2.set_xlabel("Time [s]", fontsize=12)
ax2.set_ylabel("Absolute Error [rad]", fontsize=12)
ax2.set_title("PINN Approximation Error", fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/pendulum_comparison.png", dpi=150, bbox_inches="tight")
print(f"Plot saved as: outputs/pendulum_comparison.png")

# Print statistics
print("\n" + "=" * 60)
print("STATISTICS")
print("=" * 60)
print(
    f"Mean absolute error:  {np.mean(error):.6f} rad ({np.degrees(np.mean(error)):.4f}°)"
)
print(
    f"Max absolute error:   {np.max(error):.6f} rad ({np.degrees(np.max(error)):.4f}°)"
)
print(
    f"RMSE:                 {np.sqrt(np.mean(error**2)):.6f} rad ({np.degrees(np.sqrt(np.mean(error**2))):.4f}°)"
)
print(f"Mean relative error:  {np.mean(error/np.abs(theta_true + 1e-10))*100:.2f}%")
print("=" * 60)

plt.show()
