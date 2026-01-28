# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Plot results of Damped Elastic Pendulum PINN training
"""

import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt


def find_latest_results():
    """Find the latest training output directory"""
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        print("❌ No outputs directory found. Run damped_elastic_pendulum_solver.py first!")
        return None
    
    # Find all directories with validator results
    result_dirs = []
    for root, dirs, files in os.walk(outputs_dir):
        if "validators" in dirs and "validator.npz" in os.listdir(os.path.join(root, "validators")):
            result_dirs.append(root)
    
    if not result_dirs:
        print("❌ No validator results found!")
        return None
    
    # Return the most recent one
    return max(result_dirs, key=os.path.getctime)


def analyze_results(validator_path):
    """Analyze PINN predictions vs numerical solution"""
    print(f"\n📊 Analyzing results from: {validator_path}\n")
    
    # Load data
    data = np.load(validator_path, allow_pickle=True)
    results = data['arr_0'].item()
    
    t_data = results['t'].flatten()
    theta_true = results['true_theta'].flatten()
    theta_pred = results['pred_theta'].flatten()
    
    print("=" * 70)
    print("DAMPED ELASTIC PENDULUM - PINN VALIDATION RESULTS")
    print("=" * 70)
    
    print(f"\n⏱️  Time range: [{t_data.min():.2f}, {t_data.max():.2f}] seconds")
    print(f"\n📈 Prediction statistics:")
    print(f"   θ_pred range: [{theta_pred.min():.6f}, {theta_pred.max():.6f}] rad")
    print(f"   θ_true range: [{theta_true.min():.6f}, {theta_true.max():.6f}] rad")
    
    # Check initial conditions
    t_zero_idx = np.argmin(np.abs(t_data))
    print(f"\n✓ Initial Conditions (t ≈ {t_data[t_zero_idx]:.3f}s):")
    print(f"   θ(0) predicted: {theta_pred[t_zero_idx]:10.6f} rad")
    print(f"   θ(0) expected:  {theta_true[t_zero_idx]:10.6f} rad")
    print(f"   Error: {abs(theta_pred[t_zero_idx] - theta_true[t_zero_idx]):.6e} rad")
    
    # Error statistics
    error = np.abs(theta_pred - theta_true)
    print(f"\n📉 Absolute Error Statistics:")
    print(f"   Min error:  {error.min():.6e} rad")
    print(f"   Max error:  {error.max():.6e} rad")
    print(f"   Mean error: {error.mean():.6e} rad")
    print(f"   Std error:  {error.std():.6e} rad")
    
    # Relative error
    rel_error = error / (np.abs(theta_true) + 1e-10)
    print(f"\n📊 Relative Error Statistics:")
    print(f"   Mean rel error: {np.mean(rel_error[theta_true != 0]):.4%}")
    print(f"   Max rel error:  {np.max(rel_error[theta_true != 0]):.4%}")
    
    # Check damping behavior
    dtheta = np.diff(theta_pred)
    dt = np.diff(t_data)
    velocity = dtheta / dt
    print(f"\n🌊 Damping Behavior:")
    print(f"   Velocity mean: {np.mean(velocity):.6f} rad/s")
    print(f"   Velocity std:  {np.std(velocity):.6f} rad/s")
    print(f"   Velocity range: [{velocity.min():.6f}, {velocity.max():.6f}] rad/s")
    
    # Check if damping is working (amplitude should decay)
    mid_idx = len(theta_pred) // 2
    early_amplitude = np.abs(theta_pred[:mid_idx]).max()
    late_amplitude = np.abs(theta_pred[mid_idx:]).max()
    decay_ratio = late_amplitude / (early_amplitude + 1e-10)
    print(f"   Early amplitude: {early_amplitude:.6f} rad")
    print(f"   Late amplitude:  {late_amplitude:.6f} rad")
    print(f"   Decay ratio (should be < 1): {decay_ratio:.4f} " + 
          ("✓" if decay_ratio < 1.0 else "❌"))
    
    print("\n" + "=" * 70)
    print("✓ Analysis complete!" if error.mean() < 0.01 else "⚠ Error may need improvement")
    print("=" * 70 + "\n")
    
    return t_data, theta_true, theta_pred, error


def plot_results(t_data, theta_true, theta_pred, error):
    """Create visualization plots of predictions vs ground truth"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Damped Elastic Pendulum - PINN Predictions vs Ground Truth', fontsize=16, fontweight='bold')
    
    # Plot 1: Predictions vs True Values
    ax = axes[0, 0]
    ax.plot(t_data, theta_true, 'b-', label='Ground Truth', linewidth=2, alpha=0.7)
    ax.plot(t_data, theta_pred, 'r--', label='PINN Prediction', linewidth=2, alpha=0.7)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('θ (rad)', fontsize=11)
    ax.set_title('Angle: Predicted vs True Values', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Absolute Error
    ax = axes[0, 1]
    ax.semilogy(t_data, error, 'g-', linewidth=2)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Absolute Error (rad)', fontsize=11)
    ax.set_title('Prediction Error Over Time (log scale)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    
    # Plot 3: Damping Behavior (Amplitude Envelope)
    ax = axes[1, 0]
    # Calculate envelope using maximum in rolling windows
    window_size = max(10, len(theta_pred) // 50)
    upper_envelope = np.array([np.max(np.abs(theta_pred[max(0, i-window_size):min(len(theta_pred), i+window_size)])) 
                               for i in range(len(theta_pred))])
    lower_envelope = -upper_envelope
    
    ax.plot(t_data, theta_pred, 'r-', label='PINN Prediction', linewidth=1.5, alpha=0.6)
    ax.plot(t_data, upper_envelope, 'k--', label='Amplitude Envelope', linewidth=2, alpha=0.8)
    ax.plot(t_data, lower_envelope, 'k--', linewidth=2, alpha=0.8)
    ax.fill_between(t_data, upper_envelope, lower_envelope, alpha=0.1, color='gray')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('θ (rad)', fontsize=11)
    ax.set_title('Damping Behavior - Amplitude Envelope', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Phase Space (θ vs dθ/dt)
    ax = axes[1, 1]
    dtheta_dt = np.gradient(theta_pred, t_data)
    dtheta_dt_true = np.gradient(theta_true, t_data)
    
    ax.plot(theta_true, dtheta_dt_true, 'b-', label='Ground Truth', linewidth=2, alpha=0.6)
    ax.plot(theta_pred, dtheta_dt, 'r-', label='PINN Prediction', linewidth=2, alpha=0.6)
    ax.set_xlabel('θ (rad)', fontsize=11)
    ax.set_ylabel('dθ/dt (rad/s)', fontsize=11)
    ax.set_title('Phase Space Diagram', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path("outputs") / "damped_elastic_pendulum_solver" / "validation_plots.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plots saved to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    results_dir = find_latest_results()
    if results_dir:
        validator_path = os.path.join(results_dir, "validators", "validator.npz")
        t_data, theta_true, theta_pred, error = analyze_results(validator_path)
        plot_results(t_data, theta_true, theta_pred, error)
