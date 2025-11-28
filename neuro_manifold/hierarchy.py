"""
Hierarchical Neuro-Manifold

This module defines the multi-scale structure of the system.
Instead of a single flat layer of cells, we have:
    - Micro-Cells (Sensory/Fast): High frequency updates, tied to raw input.
    - Macro-Cells (Concept/Slow): Low frequency updates, integrate information from Micro-Cells.
    - Top-Down Modulation: Macro-Cells bias the dynamics of Micro-Cells.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from .automata import ManifoldAutomata
from .geometry import RiemannianManifold
from .energy import EnergyFunction

class HierarchicalManifold(nn.Module):
    def __init__(self, input_dim: int, num_micro: int = 64, num_macro: int = 16, state_dim: int = 32):
        super().__init__()

        self.num_micro = num_micro
        self.num_macro = num_macro
        self.state_dim = state_dim

        # Geometry is shared
        self.geometry = RiemannianManifold(dim=state_dim)

        # Energy Function (Global)
        self.energy_fn = EnergyFunction(state_dim)

        # Layer 1: Sensory (Micro)
        self.micro_layer = ManifoldAutomata(num_micro, state_dim, self.geometry)

        # Layer 2: Concept (Macro)
        self.macro_layer = ManifoldAutomata(num_macro, state_dim, self.geometry)

        # Bridges
        self.bottom_up = nn.MultiheadAttention(state_dim, num_heads=2, batch_first=True)
        self.top_down = nn.Linear(state_dim, state_dim)

        self.norm = nn.LayerNorm(state_dim)

    def forward(self,
                micro_states: torch.Tensor,
                macro_states: Optional[torch.Tensor] = None,
                micro_trace: Optional[torch.Tensor] = None,
                macro_trace: Optional[torch.Tensor] = None,
                steps: int = 5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Interleaved execution of Micro and Macro layers.
        Returns: micro_states, macro_states, micro_trace, macro_trace, energy
        """
        B = micro_states.shape[0]

        if macro_states is None:
            macro_states = torch.zeros(B, self.num_macro, self.state_dim, device=micro_states.device)

        final_energy = torch.zeros(B, device=micro_states.device)

        # Simulation Loop
        for t in range(steps):
            # 1. Micro Update (Fast)
            # Apply top-down bias from previous macro state
            # Simple broadcasting: Average macro state -> Bias micro
            macro_bias = self.top_down(macro_states.mean(dim=1, keepdim=True))
            micro_states = micro_states + 0.1 * macro_bias

            micro_states, micro_trace = self.micro_layer(micro_states, hebbian_trace=micro_trace, steps=1)

            # 2. Bottom-Up Integration
            # Macro cells attend to Micro cells
            # Query: Macro, Key/Val: Micro
            integrated, _ = self.bottom_up(macro_states, micro_states, micro_states)
            macro_states = self.norm(macro_states + integrated)

            # 3. Macro Update (Slow)
            macro_states, macro_trace = self.macro_layer(macro_states, hebbian_trace=macro_trace, steps=1)

            # Calculate Energy (Thought as Equilibrium)
            # We treat energy as a monitoring signal for now, or part of intrinsic motivation
            final_energy = self.energy_fn(micro_states)

        return micro_states, macro_states, micro_trace, macro_trace, final_energy
