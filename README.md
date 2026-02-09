# PhysiGen-WM: Physically Consistent World Models

PhysiGen-WM is a research framework for integrating Lagrangian dynamics and 3D Gaussian Splatting into generative world models.

## 🚀 Key Research Innovations

### 1. Continuous Lagrangian Manifolds (Neural ODE)
Unlike discrete Seq2Seq models, PhysiGen-WM models the infinitesimal phase-space flow using a **4th-order Runge-Kutta (RK4)** solver. This ensures that the generated 3D world maintains physical continuity even at arbitrary temporal resolutions.

### 2. Variance-Constrained Optimization (VCO)
To stabilize the training of Physics-Informed Neural Networks (PINNs), we introduce a **VCO Loss** term:

$$
\mathcal{L}_{VCO} = \text{Var}(\hat{z}_{t+1} - \hat{z}_t)
$$

By minimizing the variance of the residuals, we suppress high-frequency numerical oscillations and enforce smoother gradient flow during the discovery of Hamiltonian gradients.

### 3. Spatial-Relational Attention (4-head)
We employ a multi-head attention mechanism with **Conflict-Resolved Gating** to model the geometric dependencies between Gaussian primitives. This prevents structural "collapse" or "fragmentation" during high-velocity transitions.

## 📺 Demo: Latent Physics Evolution
![PhysiGen Evolution](docs/assets/physigen_evolution.gif)
*A preview of 3D Gaussian collectives evolving via the internal Lagrangian ODE solver (Untrained Prototype).*

## 🔮 Generative Capabilities (Text-to-3D World)
![Generative World Preview](docs/assets/generative_world_preview.png)
*Initial output of the Gaussian Generative Head driven by latent physics.*

## 🛠 Project Structure
- `src/models/`: Lagrangian ODE Solvers, Gaussian Heads, and Text Adapters.
- `src/trainer.py`: Multi-GPU ready training logic with VCO and Hamiltonian loss.
- `src/utils/visualizer.py`: Utility for exporting Gaussian parameters to `.ply` format.
- `configs/`: YAML-based hyperparameter management.

## Getting Started
1. Clone to high-performance server (A100/H100 recommended).
2. Install dependencies: `pip install torch torchvision yaml tqdm`
3. Run training: `python src/trainer.py --config configs/default.yaml`

---
*Last Academic Update: 2026-02-09 23:45 (Singapore)*
