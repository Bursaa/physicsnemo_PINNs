# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Plot results of Encoder-Based Inverse PINN for Damped Pendulum
"""

import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt


def find_latest_results():
    """Find encoder results"""
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        print("❌ No outputs directory found!")
        return None

    encoder_dir = (
        outputs_dir
        / "damped_elastic_pendulum_encoder_solver/outputs/damped_elastic_pendulum_encoder_solver/"
    )
    if encoder_dir.exists():
        results_file = (
            encoder_dir
            / "encoder_results.npz"
        )
        if results_file.exists():
            return str(encoder_dir)

    print("❌ No encoder results found!")
    return None


def main():
    result_dir = find_latest_results()
    if result_dir is None:
        return
    
    results_path = os.path.join(result_dir, "encoder_results.npz")
    print(f"\n📊 Analyzing encoder-based inverse PINN results from: {result_dir}\n")
    
    data = np.load(results_path)
    
    # Extract data
    train_pred_zeta = data['train_pred_zeta']
    train_pred_theta0 = data['train_pred_theta0']
    train_pred_omega0 = data['train_pred_omega0']
    train_true_zeta = data['train_true_zeta']
    train_true_theta0 = data['train_true_theta0']
    train_true_omega0 = data['train_true_omega0']
    
    test_pred_zeta = data['test_pred_zeta']
    test_pred_theta0 = data['test_pred_theta0']
    test_pred_omega0 = data['test_pred_omega0']
    test_true_zeta = data['test_true_zeta']
    test_true_theta0 = data['test_true_theta0']
    test_true_omega0 = data['test_true_omega0']
    
    zeta_range = data['zeta_range']
    theta0_range = data['theta0_range']
    omega0_range = data['omega0_range']
    
    n_training_cases = int(data['n_training_cases'])
    n_test_cases = int(data['n_test_cases'])
    
    # Compute errors
    train_zeta_mae = np.mean(np.abs(train_pred_zeta - train_true_zeta))
    train_theta0_mae = np.mean(np.abs(train_pred_theta0 - train_true_theta0))
    train_omega0_mae = np.mean(np.abs(train_pred_omega0 - train_true_omega0))
    
    test_zeta_mae = np.mean(np.abs(test_pred_zeta - test_true_zeta))
    test_theta0_mae = np.mean(np.abs(test_pred_theta0 - test_true_theta0))
    test_omega0_mae = np.mean(np.abs(test_pred_omega0 - test_true_omega0))
    
    # Print statistics
    print("="*75)
    print("ENCODER-BASED INVERSE PINN - PARAMETER ESTIMATION RESULTS")
    print("="*75)
    
    print(f"\nConfiguration:")
    print(f"  Training cases: {n_training_cases}")
    print(f"  Test cases:     {n_test_cases}")
    print(f"  ζ range:  [{zeta_range[0]:.3f}, {zeta_range[1]:.3f}]")
    print(f"  θ₀ range: [{theta0_range[0]:.3f}, {theta0_range[1]:.3f}] rad")
    print(f"  ω₀ range: [{omega0_range[0]:.3f}, {omega0_range[1]:.3f}] rad/s")
    
    print(f"\n📈 Training Set Statistics:")
    print(f"  ζ  MAE: {train_zeta_mae:.6f}")
    print(f"  θ₀ MAE: {train_theta0_mae:.6f}")
    print(f"  ω₀ MAE: {train_omega0_mae:.6f}")
    
    print(f"\n📊 Test Set Statistics (Unseen Cases):")
    print(f"  ζ  MAE: {test_zeta_mae:.6f}")
    print(f"  θ₀ MAE: {test_theta0_mae:.6f}")
    print(f"  ω₀ MAE: {test_omega0_mae:.6f}")
    
    print("\n" + "="*75)
    
    # Create figure with parity plots
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    def parity_plot(ax, true_train, pred_train, true_test, pred_test, 
                    param_name, param_unit, param_range):
        ax.scatter(true_train, pred_train, alpha=0.5, s=20, c='tab:blue', 
                  label=f'Train (MAE={np.mean(np.abs(pred_train-true_train)):.4f})')
        ax.scatter(true_test, pred_test, alpha=0.8, s=40, c='tab:orange', 
                  label=f'Test (MAE={np.mean(np.abs(pred_test-true_test)):.4f})')
        
        # Diagonal line
        lims = [param_range[0], param_range[1]]
        ax.plot(lims, lims, 'k--', linewidth=2, label='Perfect')
        
        ax.set_xlabel(f'True {param_name} [{param_unit}]', fontsize=11)
        ax.set_ylabel(f'Predicted {param_name} [{param_unit}]', fontsize=11)
        ax.set_title(f'{param_name} Parity Plot', fontsize=12)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal', adjustable='box')
    
    # Row 1: Parity plots
    parity_plot(axes[0, 0], train_true_zeta, train_pred_zeta, 
                test_true_zeta, test_pred_zeta, 'ζ', '-', zeta_range)
    
    parity_plot(axes[0, 1], train_true_theta0, train_pred_theta0,
                test_true_theta0, test_pred_theta0, 'θ₀', 'rad', theta0_range)
    
    parity_plot(axes[0, 2], train_true_omega0, train_pred_omega0,
                test_true_omega0, test_pred_omega0, 'ω₀', 'rad/s', omega0_range)
    
    # Row 2: Error distributions
    ax4 = axes[1, 0]
    errors_train = np.abs(train_pred_zeta - train_true_zeta)
    errors_test = np.abs(test_pred_zeta - test_true_zeta)
    ax4.hist(errors_train, bins=30, alpha=0.6, label='Train', color='tab:blue')
    ax4.hist(errors_test, bins=15, alpha=0.8, label='Test', color='tab:orange')
    ax4.axvline(np.mean(errors_train), color='blue', linestyle='--', linewidth=2)
    ax4.axvline(np.mean(errors_test), color='orange', linestyle='--', linewidth=2)
    ax4.set_xlabel('|ζ_pred - ζ_true|', fontsize=11)
    ax4.set_ylabel('Count', fontsize=11)
    ax4.set_title('ζ Error Distribution', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    ax5 = axes[1, 1]
    errors_train = np.abs(train_pred_theta0 - train_true_theta0)
    errors_test = np.abs(test_pred_theta0 - test_true_theta0)
    ax5.hist(errors_train, bins=30, alpha=0.6, label='Train', color='tab:blue')
    ax5.hist(errors_test, bins=15, alpha=0.8, label='Test', color='tab:orange')
    ax5.axvline(np.mean(errors_train), color='blue', linestyle='--', linewidth=2)
    ax5.axvline(np.mean(errors_test), color='orange', linestyle='--', linewidth=2)
    ax5.set_xlabel('|θ₀_pred - θ₀_true| [rad]', fontsize=11)
    ax5.set_ylabel('Count', fontsize=11)
    ax5.set_title('θ₀ Error Distribution', fontsize=12)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    ax6 = axes[1, 2]
    errors_train = np.abs(train_pred_omega0 - train_true_omega0)
    errors_test = np.abs(test_pred_omega0 - test_true_omega0)
    ax6.hist(errors_train, bins=30, alpha=0.6, label='Train', color='tab:blue')
    ax6.hist(errors_test, bins=15, alpha=0.8, label='Test', color='tab:orange')
    ax6.axvline(np.mean(errors_train), color='blue', linestyle='--', linewidth=2)
    ax6.axvline(np.mean(errors_test), color='orange', linestyle='--', linewidth=2)
    ax6.set_xlabel('|ω₀_pred - ω₀_true| [rad/s]', fontsize=11)
    ax6.set_ylabel('Count', fontsize=11)
    ax6.set_title('ω₀ Error Distribution', fontsize=12)
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Encoder-Based Inverse PINN: Parameter Estimation Results', fontsize=14, y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(result_dir, "encoder_results_plot.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Figure saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    main()
