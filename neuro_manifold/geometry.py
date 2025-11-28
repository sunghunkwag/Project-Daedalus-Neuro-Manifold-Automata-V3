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

    def get_cholesky_factor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the Cholesky factor L(x) such that G(x) = L(x) @ L(x)^T.
        """
        # x shape: (B, N, D) or (B, N, M, D) etc.
        # We process last dim D.
        original_shape = x.shape
        x_flat = x.reshape(-1, self.dim)

        # Predict raw factors
        raw = self.cholesky_field(x_flat) # (Batch*N, D*D)

        # SAFETY: Clamp raw prediction to prevent explosion before softplus
        raw = torch.clamp(raw, -5.0, 5.0)

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

    def get_metric_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the Riemannian Metric Tensor G(x) = L(x) @ L(x)^T
        Guaranteed to be Symmetric Positive Definite (SPD).
        """
        L = self.get_cholesky_factor(x)
        # G = L @ L.T
        G = torch.matmul(L, L.transpose(-1, -2))
        return G

    def compute_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Computes the Mahalanobis distance induced by the local metric G.
        Legacy method using midpoint approximation (Slow for N^2).
        """
        dx = x2 - x1 # (..., N, M, D)

        # Compute metric at midpoint
        mid = (x1 + x2) / 2 # (..., N, M, D)
        G = self.get_metric_tensor(mid) # (..., N, M, D, D)

        # Distance squared: dx^T G dx
        dx_un = dx.unsqueeze(-1) # (..., N, M, D, 1)
        G_dx = torch.matmul(G, dx_un) # (..., N, M, D, 1)
        dist_sq = torch.matmul(dx_un.transpose(-1, -2), G_dx).squeeze(-1).squeeze(-1) # (..., N, M)

        return torch.sqrt(torch.clamp(dist_sq, min=1e-6, max=1e6))

    def compute_efficient_distance(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        Optimized distance calculation for Attention (Q vs K).
        Instead of evaluating MLP N^2 times, we approximate G_mid = (G_q + G_k) / 2.
        Input:
            Q: (B, N, D)
            K: (B, M, D)
        Output:
            Dist: (B, N, M)
        Complexity: O(N) MLP evaluations instead of O(N*M).
        """
        B, N, D = Q.shape
        M = K.shape[1]

        # 1. Precompute Metrics (O(N + M))
        # G_q: (B, N, D, D)
        L_q = self.get_cholesky_factor(Q)
        G_q = torch.matmul(L_q, L_q.transpose(-1, -2))

        # G_k: (B, M, D, D)
        L_k = self.get_cholesky_factor(K)
        G_k = torch.matmul(L_k, L_k.transpose(-1, -2))

        # 2. Compute Distances
        # d^2(q, k) approx (q-k)^T * (G_q + G_k)/2 * (q-k)
        # = 0.5 * [ (q-k)^T G_q (q-k) + (q-k)^T G_k (q-k) ]

        # Let's perform algebraic expansion for memory efficiency (avoid N*M*D*D)
        # However, for small N (e.g. < 64), direct broadcasting is faster/easier.
        # Given current settings (N=8, M=8), we use broadcasting.

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

class GeodesicFlow(nn.Module):
    """
    Simulates the flow of a particle (information) along a geodesic.
    Used for long-range communication between cells without direct edges.
    """
    def __init__(self, manifold: RiemannianManifold, step_size: float = 0.1):
        super().__init__()
        self.manifold = manifold
        self.step_size = step_size

    def forward(self, x: torch.Tensor, v: torch.Tensor, steps: int = 3) -> torch.Tensor:
        """
        Euler integration of the Geodesic Equation approx (Natural Gradient Flow).
        v_{new} = G^{-1} v_{old}
        """
        curr_x = x
        curr_v = v

        for _ in range(steps):
            L = self.manifold.get_cholesky_factor(curr_x)
            v_in = curr_v.unsqueeze(-1)

            try:
                # cholesky_solve(b, L) solves A x = b
                v_out = torch.cholesky_solve(v_in, L)
                curr_v = v_out.squeeze(-1)
            except RuntimeError:
                pass

            curr_x = curr_x + curr_v * self.step_size

        return curr_x
