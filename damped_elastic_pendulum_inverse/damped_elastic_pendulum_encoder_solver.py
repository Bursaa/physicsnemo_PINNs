# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Encoder-Based Inverse PINN Solver for Damped Pendulum

This is an advanced version that uses an encoder network to:
1. Take a time series of observations as input
2. Produce an embedding that captures the dynamics
3. Use this embedding to predict parameters (zeta, theta0, omega0)

This allows the model to generalize to completely new, unseen cases.

Architecture:
1. Encoder: Takes time series θ(t₁), θ(t₂), ..., θ(tₙ) → embedding
2. Parameter decoder: embedding → (zeta, theta0, omega0)
3. Theta network: (t, embedding) → θ(t)
4. Physics constraint: ODE residual should be zero
"""

import torch
import torch.nn as nn
import numpy as np
from sympy import Symbol
import os

import physicsnemo.sym
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_1d import Point1D
from physicsnemo.sym.domain.constraint import PointwiseConstraint
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.models.arch import Arch

from damped_elastic_pendulum_inverse import DampedPendulumInverse


class TimeSeriesEncoder(nn.Module):
    """
    Encodes a time series of theta observations into a fixed-size embedding.
    Uses 1D convolutions followed by pooling for efficient processing.
    """
    
    def __init__(self, n_time_points: int, embedding_dim: int = 64, hidden_channels: int = 64):
        super().__init__()
        
        self.n_time_points = n_time_points
        self.embedding_dim = embedding_dim
        
        # 1D convolutional encoder
        self.conv_layers = nn.Sequential(
            nn.Conv1d(2, hidden_channels, kernel_size=5, padding=2),  # Input: (t, theta)
            nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
            nn.SiLU(),
        )
        
        # Attention pooling layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_channels, 1),
            nn.Softmax(dim=1)
        )
        
        # Final embedding projection
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
    
    def forward(self, t_series, theta_series):
        """
        Args:
            t_series: (batch, n_time_points, 1) - time values
            theta_series: (batch, n_time_points, 1) - theta observations
        Returns:
            embedding: (batch, embedding_dim)
        """
        batch_size = t_series.shape[0]
        
        # Stack t and theta as channels: (batch, 2, n_time_points)
        x = torch.cat([t_series, theta_series], dim=-1)  # (batch, n_time_points, 2)
        x = x.transpose(1, 2)  # (batch, 2, n_time_points)
        
        # Apply convolutions
        x = self.conv_layers(x)  # (batch, hidden_channels, n_time_points)
        
        # Transpose for attention: (batch, n_time_points, hidden_channels)
        x = x.transpose(1, 2)
        
        # Attention pooling
        attn_weights = self.attention(x)  # (batch, n_time_points, 1)
        x = torch.sum(x * attn_weights, dim=1)  # (batch, hidden_channels)
        
        # Final embedding
        embedding = self.fc(x)  # (batch, embedding_dim)
        
        return embedding


class EncoderBasedParameterNet(Arch):
    """
    Network that combines encoder and parameter prediction.
    
    Takes time series as input and outputs physical parameters.
    Also outputs the embedding for use by the theta network.
    """
    
    def __init__(
        self,
        n_time_points: int,
        embedding_dim: int = 64,
        hidden_size: int = 128,
        zeta_range: tuple = (0.05, 1.0),
        theta0_range: tuple = (-np.pi/2, np.pi/2),
        omega0_range: tuple = (-2.0, 2.0),
    ):
        super().__init__(
            input_keys=[Key("t_series"), Key("theta_series")],
            output_keys=[Key("zeta"), Key("theta0"), Key("omega0"), Key("embedding")],
            periodicity=None,
        )
        
        self.n_time_points = n_time_points
        self.embedding_dim = embedding_dim
        self.zeta_range = zeta_range
        self.theta0_range = theta0_range
        self.omega0_range = omega0_range
        
        # Encoder for time series
        self.encoder = TimeSeriesEncoder(n_time_points, embedding_dim)
        
        # Parameter prediction heads
        self.param_decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        
        self.zeta_head = nn.Linear(hidden_size, 1)
        self.theta0_head = nn.Linear(hidden_size, 1)
        self.omega0_head = nn.Linear(hidden_size, 1)
    
    def forward(self, invar):
        t_series = invar["t_series"]  # (batch, n_time_points, 1)
        theta_series = invar["theta_series"]  # (batch, n_time_points, 1)
        
        # Encode time series
        embedding = self.encoder(t_series, theta_series)  # (batch, embedding_dim)
        
        # Decode parameters
        hidden = self.param_decoder(embedding)
        
        # Output heads with bounded outputs
        zeta_raw = self.zeta_head(hidden)
        zeta = torch.sigmoid(zeta_raw) * (self.zeta_range[1] - self.zeta_range[0]) + self.zeta_range[0]
        
        theta0_raw = self.theta0_head(hidden)
        theta0 = torch.tanh(theta0_raw) * (self.theta0_range[1] - self.theta0_range[0]) / 2 + \
                 (self.theta0_range[1] + self.theta0_range[0]) / 2
        
        omega0_raw = self.omega0_head(hidden)
        omega0 = torch.tanh(omega0_raw) * (self.omega0_range[1] - self.omega0_range[0]) / 2 + \
                 (self.omega0_range[1] + self.omega0_range[0]) / 2
        
        return {
            "zeta": zeta,
            "theta0": theta0,
            "omega0": omega0,
            "embedding": embedding,
        }


class ConditionedThetaNet(nn.Module):
    """
    Network that predicts theta(t) given time and embedding from encoder.
    """
    
    def __init__(self, embedding_dim: int, hidden_size: int = 256, num_layers: int = 6):
        super().__init__()
        
        # Input: t_norm (1) + embedding (embedding_dim)
        input_size = 1 + embedding_dim
        
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.SiLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.SiLU())
        
        layers.append(nn.Linear(hidden_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, t_norm, embedding):
        """
        Args:
            t_norm: (batch, 1) - normalized time
            embedding: (batch, embedding_dim) - case embedding
        Returns:
            theta: (batch, 1)
        """
        x = torch.cat([t_norm, embedding], dim=-1)
        return self.network(x)


def generate_training_data(n_cases, n_time_points, t_max, g, l,
                           zeta_range, theta0_range, omega0_range,
                           noise_level=0.0, seed=42):
    """
    Generate training data with full time series for each case.
    
    Returns data in the format needed for encoder-based training.
    """
    from scipy.integrate import solve_ivp
    
    np.random.seed(seed)
    
    # Sample random parameters
    zetas = np.random.uniform(zeta_range[0], zeta_range[1], n_cases)
    theta0s = np.random.uniform(theta0_range[0], theta0_range[1], n_cases)
    omega0s = np.random.uniform(omega0_range[0], omega0_range[1], n_cases)
    
    t_eval = np.linspace(0, t_max, n_time_points)
    
    # Store time series for each case
    t_series_all = []
    theta_series_all = []
    
    for case_idx in range(n_cases):
        zeta = zetas[case_idx]
        theta0 = theta0s[case_idx]
        omega0 = omega0s[case_idx]
        
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
        
        theta_case = sol.y[0]
        
        if noise_level > 0:
            theta_case = theta_case + noise_level * np.random.randn(len(theta_case))
        
        t_series_all.append(t_eval)
        theta_series_all.append(theta_case)
    
    # Convert to numpy arrays: (n_cases, n_time_points)
    t_series = np.array(t_series_all, dtype=np.float32)
    theta_series = np.array(theta_series_all, dtype=np.float32)
    
    true_params = {
        "zeta": zetas.astype(np.float32),
        "theta0": theta0s.astype(np.float32),
        "omega0": omega0s.astype(np.float32),
    }
    
    return t_series, theta_series, true_params


class EncoderInversePINNTrainer:
    """
    Custom trainer for encoder-based inverse PINN.
    
    Since the standard PhysicsNeMo solver expects fixed input shapes,
    we implement custom training for the encoder-based approach.
    """
    
    def __init__(self, encoder_param_net, theta_net, physics_nodes, 
                 t_max, g, l, device='cuda'):
        self.encoder_param_net = encoder_param_net.to(device)
        self.theta_net = theta_net.to(device)
        self.physics_nodes = physics_nodes
        self.t_max = t_max
        self.g = g
        self.l = l
        self.device = device
        
        # Optimizer for all parameters
        self.optimizer = torch.optim.Adam(
            list(encoder_param_net.parameters()) + list(theta_net.parameters()),
            lr=0.001
        )
        
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9995)
    
    def compute_theta_derivatives(self, theta, t):
        """Compute first and second derivatives of theta w.r.t. time"""
        theta_t = torch.autograd.grad(
            theta, t, grad_outputs=torch.ones_like(theta),
            create_graph=True, retain_graph=True
        )[0]
        
        theta_tt = torch.autograd.grad(
            theta_t, t, grad_outputs=torch.ones_like(theta_t),
            create_graph=True, retain_graph=True
        )[0]
        
        return theta_t, theta_tt
    
    def physics_residual(self, theta, theta_t, theta_tt, zeta):
        """Compute ODE residual: θ'' + ζθ' + (g/l)sin(θ) = 0"""
        return theta_tt + zeta * theta_t + (self.g / self.l) * torch.sin(theta)
    
    def train_step(self, t_series, theta_series, true_params, 
                   lambda_data=100.0, lambda_physics=1.0, lambda_ic=100.0,
                   n_interior_per_case=50):
        """
        Single training step.
        
        Args:
            t_series: (batch, n_time_points) - time values for each case
            theta_series: (batch, n_time_points) - observations for each case
            true_params: dict with true zeta, theta0, omega0 (for supervised loss)
        """
        batch_size, n_time_points = t_series.shape
        
        self.optimizer.zero_grad()
        
        # Move data to device
        t_series_tensor = torch.tensor(t_series, device=self.device).unsqueeze(-1)  # (batch, n_time_points, 1)
        theta_series_tensor = torch.tensor(theta_series, device=self.device).unsqueeze(-1)
        
        # Encode time series and predict parameters
        encoder_input = {
            "t_series": t_series_tensor,
            "theta_series": theta_series_tensor
        }
        encoder_output = self.encoder_param_net(encoder_input)
        
        pred_zeta = encoder_output["zeta"]  # (batch, 1)
        pred_theta0 = encoder_output["theta0"]
        pred_omega0 = encoder_output["omega0"]
        embedding = encoder_output["embedding"]  # (batch, embedding_dim)
        
        # ==========================================================================
        # LOSS 1: Parameter regression loss (supervised)
        # ==========================================================================
        true_zeta = torch.tensor(true_params["zeta"], device=self.device).unsqueeze(-1)
        true_theta0 = torch.tensor(true_params["theta0"], device=self.device).unsqueeze(-1)
        true_omega0 = torch.tensor(true_params["omega0"], device=self.device).unsqueeze(-1)
        
        param_loss = (
            torch.mean((pred_zeta - true_zeta)**2) +
            torch.mean((pred_theta0 - true_theta0)**2) +
            torch.mean((pred_omega0 - true_omega0)**2)
        )
        
        # ==========================================================================
        # LOSS 2: Data fitting loss
        # ==========================================================================
        # For each case, predict theta at observation times
        data_loss = 0.0
        for case_idx in range(batch_size):
            t_case = t_series_tensor[case_idx]  # (n_time_points, 1)
            theta_true_case = theta_series_tensor[case_idx]  # (n_time_points, 1)
            emb_case = embedding[case_idx:case_idx+1].expand(n_time_points, -1)  # (n_time_points, emb_dim)
            
            t_norm = t_case / self.t_max
            theta_pred = self.theta_net(t_norm, emb_case)
            
            data_loss += torch.mean((theta_pred - theta_true_case)**2)
        
        data_loss = data_loss / batch_size
        
        # ==========================================================================
        # LOSS 3: Physics loss (ODE residual)
        # ==========================================================================
        physics_loss = 0.0
        for case_idx in range(batch_size):
            # Sample interior points
            t_interior = torch.linspace(0.001, self.t_max, n_interior_per_case, 
                                        device=self.device).unsqueeze(-1)
            t_interior.requires_grad_(True)
            
            emb_case = embedding[case_idx:case_idx+1].expand(n_interior_per_case, -1)
            zeta_case = pred_zeta[case_idx:case_idx+1].expand(n_interior_per_case, -1)
            
            t_norm = t_interior / self.t_max
            theta = self.theta_net(t_norm, emb_case)
            
            # Compute derivatives
            theta_t, theta_tt = self.compute_theta_derivatives(theta, t_interior)
            
            # Compute residual
            residual = self.physics_residual(theta, theta_t, theta_tt, zeta_case)
            physics_loss += torch.mean(residual**2)
        
        physics_loss = physics_loss / batch_size
        
        # ==========================================================================
        # LOSS 4: Initial condition loss
        # ==========================================================================
        ic_loss = 0.0
        for case_idx in range(batch_size):
            t_ic = torch.zeros(1, 1, device=self.device, requires_grad=True)
            emb_case = embedding[case_idx:case_idx+1]
            
            t_norm_ic = t_ic / self.t_max
            theta_ic = self.theta_net(t_norm_ic, emb_case)
            
            # Compute derivative at t=0
            theta_t_ic = torch.autograd.grad(
                theta_ic, t_ic, grad_outputs=torch.ones_like(theta_ic),
                create_graph=True
            )[0] / self.t_max  # Chain rule correction
            
            ic_loss += (theta_ic - pred_theta0[case_idx])**2
            ic_loss += (theta_t_ic - pred_omega0[case_idx])**2
        
        ic_loss = ic_loss.mean() / batch_size
        
        # ==========================================================================
        # Total loss
        # ==========================================================================
        total_loss = (
            lambda_data * data_loss +
            lambda_physics * physics_loss +
            lambda_ic * ic_loss +
            10.0 * param_loss  # Strong supervision on parameters
        )
        
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            "total": total_loss.item(),
            "data": data_loss.item(),
            "physics": physics_loss.item(),
            "ic": ic_loss.item(),
            "param": param_loss.item(),
        }
    
    def evaluate(self, t_series, theta_series, true_params):
        """Evaluate model on test cases."""
        self.encoder_param_net.eval()
        self.theta_net.eval()
        
        with torch.no_grad():
            t_series_tensor = torch.tensor(t_series, device=self.device).unsqueeze(-1)
            theta_series_tensor = torch.tensor(theta_series, device=self.device).unsqueeze(-1)
            
            encoder_input = {
                "t_series": t_series_tensor,
                "theta_series": theta_series_tensor
            }
            encoder_output = self.encoder_param_net(encoder_input)
            
            pred_zeta = encoder_output["zeta"].cpu().numpy().flatten()
            pred_theta0 = encoder_output["theta0"].cpu().numpy().flatten()
            pred_omega0 = encoder_output["omega0"].cpu().numpy().flatten()
        
        self.encoder_param_net.train()
        self.theta_net.train()
        
        return {
            "pred_zeta": pred_zeta,
            "pred_theta0": pred_theta0,
            "pred_omega0": pred_omega0,
            "true_zeta": true_params["zeta"],
            "true_theta0": true_params["theta0"],
            "true_omega0": true_params["omega0"],
            "zeta_mae": np.mean(np.abs(pred_zeta - true_params["zeta"])),
            "theta0_mae": np.mean(np.abs(pred_theta0 - true_params["theta0"])),
            "omega0_mae": np.mean(np.abs(pred_omega0 - true_params["omega0"])),
        }


@physicsnemo.sym.main(config_path="conf", config_name="config_encoder")
def run(cfg: PhysicsNeMoConfig) -> None:
    # ==========================================================================
    # CONFIGURATION
    # ==========================================================================
    g = 9.81
    l = 1.0
    t_max = 20.0
    
    zeta_range = (0.05, 0.8)
    theta0_range = (-np.pi/3, np.pi/3)
    omega0_range = (-1.5, 1.5)
    
    n_training_cases = cfg.custom.n_training_cases
    n_test_cases = cfg.custom.n_test_cases
    n_time_points = cfg.custom.n_time_points
    noise_level = cfg.custom.noise_level
    embedding_dim = cfg.custom.embedding_dim
    max_steps = cfg.training.max_steps
    batch_size = cfg.custom.encoder_batch_size
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("\n" + "="*70)
    print("ENCODER-BASED INVERSE PINN FOR DAMPED PENDULUM")
    print("="*70)
    print(f"\nDevice: {device}")
    print(f"Training cases: {n_training_cases}")
    print(f"Test cases: {n_test_cases}")
    print(f"Time points per case: {n_time_points}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Noise level: {noise_level}")
    print(f"Max training steps: {max_steps}")
    print("="*70 + "\n")

    # ==========================================================================
    # GENERATE DATA
    # ==========================================================================
    print("Generating training data...")
    t_train, theta_train, true_params_train = generate_training_data(
        n_training_cases, n_time_points, t_max, g, l,
        zeta_range, theta0_range, omega0_range,
        noise_level=noise_level, seed=42
    )
    print(f"Training data shape: t={t_train.shape}, theta={theta_train.shape}")
    
    print("Generating test data...")
    t_test, theta_test, true_params_test = generate_training_data(
        n_test_cases, n_time_points, t_max, g, l,
        zeta_range, theta0_range, omega0_range,
        noise_level=0.0, seed=999
    )
    print(f"Test data shape: t={t_test.shape}, theta={theta_test.shape}")

    # ==========================================================================
    # SETUP NETWORKS
    # ==========================================================================
    encoder_param_net = EncoderBasedParameterNet(
        n_time_points=n_time_points,
        embedding_dim=embedding_dim,
        hidden_size=128,
        zeta_range=zeta_range,
        theta0_range=theta0_range,
        omega0_range=omega0_range,
    )
    
    theta_net = ConditionedThetaNet(
        embedding_dim=embedding_dim,
        hidden_size=cfg.arch.fully_connected.layer_size,
        num_layers=cfg.arch.fully_connected.nr_layers,
    )
    
    # Initialize trainer
    trainer = EncoderInversePINNTrainer(
        encoder_param_net, theta_net, None, t_max, g, l, device
    )

    # ==========================================================================
    # TRAINING LOOP
    # ==========================================================================
    print("\nStarting training...")
    
    n_batches = (n_training_cases + batch_size - 1) // batch_size
    
    for step in range(max_steps):
        # Random batch selection
        batch_indices = np.random.choice(n_training_cases, size=batch_size, replace=False)
        
        t_batch = t_train[batch_indices]
        theta_batch = theta_train[batch_indices]
        params_batch = {
            "zeta": true_params_train["zeta"][batch_indices],
            "theta0": true_params_train["theta0"][batch_indices],
            "omega0": true_params_train["omega0"][batch_indices],
        }
        
        losses = trainer.train_step(
            t_batch, theta_batch, params_batch,
            lambda_data=cfg.custom.lambda_data,
            lambda_physics=cfg.custom.lambda_physics,
            lambda_ic=cfg.custom.lambda_ic,
            n_interior_per_case=cfg.custom.n_interior_per_case,
        )
        
        if step % 500 == 0:
            print(f"Step {step:5d} | Loss: {losses['total']:.6f} | "
                  f"Data: {losses['data']:.6f} | Physics: {losses['physics']:.6f} | "
                  f"IC: {losses['ic']:.6f} | Param: {losses['param']:.6f}")
        
        # Evaluation
        if step % 2000 == 0:
            train_eval = trainer.evaluate(t_train, theta_train, true_params_train)
            test_eval = trainer.evaluate(t_test, theta_test, true_params_test)
            
            print(f"\n  Training MAE - ζ: {train_eval['zeta_mae']:.6f}, "
                  f"θ₀: {train_eval['theta0_mae']:.6f}, ω₀: {train_eval['omega0_mae']:.6f}")
            print(f"  Test MAE     - ζ: {test_eval['zeta_mae']:.6f}, "
                  f"θ₀: {test_eval['theta0_mae']:.6f}, ω₀: {test_eval['omega0_mae']:.6f}\n")

    # ==========================================================================
    # FINAL EVALUATION
    # ==========================================================================
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    train_eval = trainer.evaluate(t_train, theta_train, true_params_train)
    test_eval = trainer.evaluate(t_test, theta_test, true_params_test)
    
    print("\n--- Training Set ---")
    print(f"{'Metric':<15} {'ζ':<15} {'θ₀':<15} {'ω₀':<15}")
    print("-"*60)
    print(f"{'MAE':<15} {train_eval['zeta_mae']:<15.6f} "
          f"{train_eval['theta0_mae']:<15.6f} {train_eval['omega0_mae']:<15.6f}")
    
    print("\n--- Test Set (Unseen Cases) ---")
    print(f"{'Metric':<15} {'ζ':<15} {'θ₀':<15} {'ω₀':<15}")
    print("-"*60)
    print(f"{'MAE':<15} {test_eval['zeta_mae']:<15.6f} "
          f"{test_eval['theta0_mae']:<15.6f} {test_eval['omega0_mae']:<15.6f}")
    
    # Show individual test case predictions
    print("\n--- Individual Test Case Predictions ---")
    print(f"{'Case':<6} {'True ζ':<10} {'Pred ζ':<10} {'True θ₀':<10} {'Pred θ₀':<10} {'True ω₀':<10} {'Pred ω₀':<10}")
    print("-"*66)
    for i in range(min(n_test_cases, 10)):
        print(f"{i:<6} {test_eval['true_zeta'][i]:<10.4f} {test_eval['pred_zeta'][i]:<10.4f} "
              f"{test_eval['true_theta0'][i]:<10.4f} {test_eval['pred_theta0'][i]:<10.4f} "
              f"{test_eval['true_omega0'][i]:<10.4f} {test_eval['pred_omega0'][i]:<10.4f}")
    
    print("\n" + "="*70)
    
    # Save results
    output_dir = "outputs/damped_elastic_pendulum_encoder_solver"
    os.makedirs(output_dir, exist_ok=True)
    
    np.savez(
        f"{output_dir}/encoder_results.npz",
        # Training results
        train_pred_zeta=train_eval['pred_zeta'],
        train_pred_theta0=train_eval['pred_theta0'],
        train_pred_omega0=train_eval['pred_omega0'],
        train_true_zeta=train_eval['true_zeta'],
        train_true_theta0=train_eval['true_theta0'],
        train_true_omega0=train_eval['true_omega0'],
        # Test results
        test_pred_zeta=test_eval['pred_zeta'],
        test_pred_theta0=test_eval['pred_theta0'],
        test_pred_omega0=test_eval['pred_omega0'],
        test_true_zeta=test_eval['true_zeta'],
        test_true_theta0=test_eval['true_theta0'],
        test_true_omega0=test_eval['true_omega0'],
        # Configuration
        zeta_range=np.array(zeta_range),
        theta0_range=np.array(theta0_range),
        omega0_range=np.array(omega0_range),
        n_training_cases=n_training_cases,
        n_test_cases=n_test_cases,
    )
    print(f"Results saved to {output_dir}/encoder_results.npz")
    
    # Save model
    torch.save({
        'encoder_param_net': encoder_param_net.state_dict(),
        'theta_net': theta_net.state_dict(),
    }, f"{output_dir}/model.pth")
    print(f"Model saved to {output_dir}/model.pth")


if __name__ == "__main__":
    run()
