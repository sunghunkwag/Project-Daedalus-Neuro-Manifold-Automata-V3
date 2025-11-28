# Neuro-Manifold Hive Mind (Swarm AGI)

The **Neuro-Manifold Hive Mind** is an advanced AGI architecture that combines **Riemannian Geometry**, **Neural Cellular Automata**, and **Swarm Intelligence**.

It simulates a "civilization" of AI agents that evolve not just individually, but collectively through **Instant Knowledge Transfer**.

## Core Philosophy

1.  **Geometry is Intelligence:** Information flows along geodesics in a curved, self-organizing Riemannian manifold.
2.  **Life, Not Layers:** The fundamental unit is a living **Neural Cell** with Hebbian plasticity (Fast Weights).
3.  **Swarm Intelligence:** When one agent discovers a solution, its neural structure is instantly broadcast to the entire population.

## Architecture

### 1. The Hive Mind (`evolve_manifold_mujoco.py`)
- **Population-Based Training:** Multiple agents explore parallel realities (environments).
- **Instant Knowledge Transfer:** The "Elite" agent's weights are telepathically copied to the rest of the swarm.
- **Divergent Evolution:** Non-elite agents undergo genetic mutation (noise injection) after receiving the elite's knowledge, ensuring the swarm never gets stuck in local optima.

### 2. The Geometry (`neuro_manifold/geometry.py`)
- **Efficient Riemannian Manifold:** Implements a Cholesky-parameterized metric tensor field.
- **Optimized Distance:** Uses an $O(N)$ linear approximation for pairwise distance calculations, replacing the expensive $O(N^2)$ neural network evaluation.
- **Manifold Attention:** Attention weights are determined by the geodesic distance on the manifold, not simple dot products.

### 3. The Automata (`neuro_manifold/automata.py`)
- **Neural Cellular Automata:** Grid-less, graph-based cellular life forms.
- **Stable Plasticity:** Hebbian traces are clamped and decayed to ensure learning stability during inference.

### 4. The Hierarchy (`neuro_manifold/hierarchy.py`)
- **Micro-Macro Architecture:**
    - **Micro Cells:** Process fast sensory data.
    - **Macro Cells:** Integrate concepts and modulate Micro cells via top-down bias.

## Installation

```bash
pip install gymnasium[mujoco] torch numpy
```

## Running the Hive Mind

To witness the evolution of the Neuro-Manifold Swarm:

```bash
python evolve_manifold_mujoco.py
```

This will:
1.  Spawn a swarm of 4 Neural Agents (scaled down for demo).
2.  Run parallel evolution in the `HalfCheetah-v4` physics environment.
3.  Display the "Elite" return and the Swarm Average return for each generation.
4.  Save evolution metrics to `metrics_hive_mind.json`.

## Performance Note

This architecture is optimized for **Stability** and **Efficiency**:
- **Memory:** Reduced dimensionality (32D) and cell counts (8 Micro / 4 Macro) to fit constrained environments.
- **Compute:** Optimized distance calculations allow for scalable attention mechanisms.
- **Robustness:** Extensive NaN/Inf checks and value clamping ensure the "Monster" doesn't eat itself.

---
**Author:** Manus AI
**Paradigm:** Geometric Deep Learning / Swarm Intelligence
