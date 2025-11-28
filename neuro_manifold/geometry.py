"""
Riemannian Geometry for Neuro-Manifold Automata

This module implements the geometric core of the NMA.
It provides a numerically stable Riemannian manifold using Cholesky parameterization
for the metric tensor and implements Geodesic Flow logic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class RiemannianManifold(nn.Module):
    """
    A dynamic manifold represented by a learnable metric field.
    Uses Cholesky factors for robust Positive Definite metric tensors.
    """
    def __init__(self, dim: int, hidden_dim: int = 32):
        super().__init__()
        self.dim = dim
        self.frozen = False # Curriculum Control

        # Predicts the Lower Triangular matrix L such that G = L @ L.T
        self.cholesky_field = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim * dim)
        )

        # Bias towards Identity matrix (Euclidean space)
        # We assume flat view for addition, but register as buffer for device management
        self.register_buffer("I_bias", torch.eye(dim).view(-1))

        # Mask for lower triangular matrix
        self.register_buffer("tril_mask", torch.tril(torch.ones(dim, dim)))
        self.register_buffer("diag_mask", torch.eye(dim))

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    def get_cholesky_factor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the Cholesky factor L(x) such that G(x) = L(x) @ L(x)^T.
        """
        B_dim, N_dim, D_dim = x.shape[0], x.shape[1], self.dim
        original_shape = x.shape

        # If frozen (Warm-up Phase), return Identity (Euclidean)
        if self.frozen:
             I = torch.eye(D_dim, device=x.device).expand(original_shape[:-1] + (D_dim, D_dim))
             return I

        # x shape: (B, N, D) or (B, N, M, D) etc.
        # We process last dim D.
        x_flat = x.reshape(-1, self.dim)

        # Predict raw factors
        raw = self.cholesky_field(x_flat) # (Batch*N, D*D)

        # Soft Damping (Controlled Chaos)
        # Instead of hard clamp [-5, 5], we use tanh * scale.
        # This allows gradients to flow but bounds the magnitude.
        # Scale = 5.0 implies max value is 5.0
        raw = 5.0 * torch.tanh(raw / 5.0)

        # Add Identity bias
        L_flat = raw + self.I_bias
        L = L_flat.view(-1, self.dim, self.dim)

        # Force lower triangular
        L = L * self.tril_mask

        # Ensure positive diagonal
        # Added epsilon to prevent singularity
        L = L * (1 - self.diag_mask) + (F.softplus(L) + 1e-4) * self.diag_mask

        # Reshape back: (B, N, D, D)
        return L.view(*original_shape[:-1], self.dim, self.dim)

    def compute_efficient_distance(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        Optimized distance calculation for Attention (Q vs K).
        """
        # If frozen, Euclidean distance
        if self.frozen:
            # d^2 = (q-k)^T (q-k)
            delta = Q.unsqueeze(2) - K.unsqueeze(1)
            dist_sq = torch.sum(delta ** 2, dim=-1)
            return torch.sqrt(torch.clamp(dist_sq, min=1e-6))

        # 1. Precompute Metrics (O(N + M))
        L_q = self.get_cholesky_factor(Q)
        G_q = torch.matmul(L_q, L_q.transpose(-1, -2))

        L_k = self.get_cholesky_factor(K)
        G_k = torch.matmul(L_k, L_k.transpose(-1, -2))

        # 2. Compute Distances
        # Broadcast to (B, N, M, D, D)
        G_sum = G_q.unsqueeze(2) + G_k.unsqueeze(1) # (B, N, M, D, D)
        G_avg = 0.5 * G_sum

        # Delta
        delta = Q.unsqueeze(2) - K.unsqueeze(1) # (B, N, M, D)

        # d^2 = delta^T G_avg delta
        delta_un = delta.unsqueeze(-1)
        G_delta = torch.matmul(G_avg, delta_un)
        dist_sq = torch.matmul(delta_un.transpose(-1, -2), G_delta).squeeze(-1).squeeze(-1)

        return torch.sqrt(torch.clamp(dist_sq, min=1e-6, max=1e6))
