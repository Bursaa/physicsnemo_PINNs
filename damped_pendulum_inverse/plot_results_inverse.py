# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Plot results of Inverse PINN for Damped Pendulum parameter estimation
"""

import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt


def find_latest_results():
    """Find the latest training output directory"""
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        print("❌ No outputs directory found. Run damped_elastic_pendulum_inverse_solver.py first!")
        return None
    
    # Find all directories with validator results
    result_dirs = []
    for root, dirs, files in os.walk(outputs_dir):
        if "validators" in dirs and "validator.npz" in os.listdir(os.path.join(root, "validators")):
            result_dirs.append(root)
    
    if not result_dirs:
        print("❌ No validator results found!")
        return None
    
    return max(result_dirs, key=os.path.getctime)


def main():
    result_dir = find_latest_results()
    if result_dir is None:
        return
    
    validator_path = os.path.join(result_dir, "validators", "validator.npz")
    params_path = os.path.join(result_dir, "estimated_parameters.npz")
    
    print(f"\n📊 Analyzing inverse PINN results from: {result_dir}\n")
    
    # Load validation data
    data = np.load(validator_path, allow_pickle=True)
    results = data['arr_0'].item()
    
    t_data = results['t'].flatten()
    theta_true = results['true_theta'].flatten()
    theta_pred = results['pred_theta'].flatten()
    
    # Load estimated parameters
    if os.path.exists(params_path):
        params = np.load(params_path)
        true_zeta = params['true_zeta']
        true_theta0 = params['true_theta0']
        true_omega0 = params['true_omega0']
        est_zeta = params['estimated_zeta']
        est_theta0 = params['estimated_theta0']
        est_omega0 = params['estimated_omega0']
        init_zeta = params['init_zeta']
        init_theta0 = params['init_theta0']
        init_omega0 = params['init_omega0']
    else:
        print("⚠️  Parameter estimates file not found!")
        true_zeta = est_zeta = init_zeta = None
        true_theta0 = est_theta0 = init_theta0 = None
        true_omega0 = est_omega0 = init_omega0 = None
    
    # Print results
    print("="*75)
    print("INVERSE PINN - DAMPED PENDULUM PARAMETER ESTIMATION RESULTS")
    print("="*75)
    
    if est_zeta is not None:
        print(f"\n{'Parameter':<15} {'True Value':<15} {'Initial Guess':<15} {'Estimated':<15} {'Rel. Error':<12}")
        print("-"*75)
        
        zeta_rel = 100 * abs(est_zeta - true_zeta) / true_zeta
        print(f"{'ζ (zeta)':<15} {true_zeta:<15.6f} {init_zeta:<15.6f} {est_zeta:<15.6f} {zeta_rel:<11.2f}%")
        
        theta0_rel = 100 * abs(est_theta0 - true_theta0) / abs(true_theta0)
        print(f"{'θ₀ [rad]':<15} {true_theta0:<15.6f} {init_theta0:<15.6f} {est_theta0:<15.6f} {theta0_rel:<11.2f}%")
        
        omega0_rel_str = f"{100*abs(est_omega0 - true_omega0):.2f}%" if true_omega0 != 0 else f"{abs(est_omega0):.4f} abs"
        print(f"{'ω₀ [rad/s]':<15} {true_omega0:<15.6f} {init_omega0:<15.6f} {est_omega0:<15.6f} {omega0_rel_str:<12}")
    
    # Error statistics
    error = np.abs(theta_pred - theta_true)
    print(f"\n📉 Solution Absolute Error Statistics:")
    print(f"   Min error:  {error.min():.6e} rad")
    print(f"   Max error:  {error.max():.6e} rad")
    print(f"   Mean error: {error.mean():.6e} rad")
    print(f"   Std error:  {error.std():.6e} rad")
    
    print("\n" + "="*75)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: θ(t) comparison
    ax1 = axes[0, 0]
    ax1.plot(t_data, theta_true, 'b-', linewidth=2, label='True solution', alpha=0.8)
    ax1.plot(t_data, theta_pred, 'r--', linewidth=2, label='PINN prediction', alpha=0.8)
    ax1.set_xlabel('Time [s]', fontsize=12)
    ax1.set_ylabel('θ [rad]', fontsize=12)
    ax1.set_title('Damped Pendulum: θ(t)', fontsize=14)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Absolute error over time
    ax2 = axes[0, 1]
    ax2.semilogy(t_data, error, 'k-', linewidth=1.5)
    ax2.set_xlabel('Time [s]', fontsize=12)
    ax2.set_ylabel('|θ_pred - θ_true| [rad]', fontsize=12)
    ax2.set_title('Absolute Error over Time', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=error.mean(), color='r', linestyle='--', label=f'Mean: {error.mean():.2e}')
    ax2.legend(fontsize=10)
    
    # Plot 3: Phase space
    ax3 = axes[1, 0]
    # Approximate velocity from numerical differentiation
    dt = t_data[1] - t_data[0]
    theta_dot_true = np.gradient(theta_true, dt)
    theta_dot_pred = np.gradient(theta_pred, dt)
    
    ax3.plot(theta_true, theta_dot_true, 'b-', linewidth=1.5, label='True', alpha=0.7)
    ax3.plot(theta_pred, theta_dot_pred, 'r--', linewidth=1.5, label='PINN', alpha=0.7)
    ax3.set_xlabel('θ [rad]', fontsize=12)
    ax3.set_ylabel('dθ/dt [rad/s]', fontsize=12)
    ax3.set_title('Phase Space', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal', adjustable='box')
    
    # Plot 4: Parameter estimation summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    if est_zeta is not None:
        summary_text = "PARAMETER ESTIMATION SUMMARY\n" + "="*40 + "\n\n"
        summary_text += f"{'Parameter':<12} {'True':<12} {'Estimated':<12} {'Error':<12}\n"
        summary_text += "-"*48 + "\n"
        summary_text += f"{'ζ (zeta)':<12} {true_zeta:<12.4f} {est_zeta:<12.4f} {abs(est_zeta-true_zeta):<12.4f}\n"
        summary_text += f"{'θ₀ [rad]':<12} {true_theta0:<12.4f} {est_theta0:<12.4f} {abs(est_theta0-true_theta0):<12.4f}\n"
        summary_text += f"{'ω₀ [rad/s]':<12} {true_omega0:<12.4f} {est_omega0:<12.4f} {abs(est_omega0-true_omega0):<12.4f}\n"
        summary_text += "\n" + "-"*48 + "\n"
        
        # Calculate overall quality
        zeta_quality = "✓" if abs(est_zeta - true_zeta) / true_zeta < 0.05 else "✗"
        theta0_quality = "✓" if abs(est_theta0 - true_theta0) / abs(true_theta0) < 0.05 else "✗"
        omega0_quality = "✓" if abs(est_omega0 - true_omega0) < 0.1 else "✗"
        
        summary_text += f"\nAccuracy (< 5% error):\n"
        summary_text += f"  ζ:  {zeta_quality}  ({100*abs(est_zeta-true_zeta)/true_zeta:.2f}%)\n"
        summary_text += f"  θ₀: {theta0_quality}  ({100*abs(est_theta0-true_theta0)/abs(true_theta0):.2f}%)\n"
        summary_text += f"  ω₀: {omega0_quality}  (error: {abs(est_omega0-true_omega0):.4f})\n"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(result_dir, "inverse_pinn_results.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plots saved to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    main()
