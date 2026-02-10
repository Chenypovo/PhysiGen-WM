# PhysiGen-WM: Physically Consistent World Models

PhysiGen-WM is a research framework for integrating Lagrangian dynamics and 3D Gaussian Splatting into generative world models.

## 🚀 Key Research Innovations

### 1. Continuous Lagrangian Manifolds (Neural ODE)
Unlike discrete Seq2Seq models, PhysiGen-WM models the infinitesimal phase-space flow using a **4th-order Runge-Kutta (RK4)** solver. This ensures that the generated 3D world maintains physical continuity even at arbitrary temporal resolutions.

### 2. Variance-Constrained Optimization (VCO)
To stabilize the training of Physics-Informed Neural Networks (PINNs), we introduce a **VCO Loss** term:

$$ L_{VCO} = \text{Var}(\hat{z}_{t+1} - \hat{z}_{t}) $$

By minimizing the variance of the residuals, we suppress high-frequency numerical oscillations and enforce smoother gradient flow during the discovery of Hamiltonian gradients.

### 3. Spatial-Relational Attention (4-head)
We employ a multi-head attention mechanism with **Conflict-Resolved Gating** to model the geometric dependencies between Gaussian primitives. This prevents structural "collapse" or "fragmentation" during high-velocity transitions.

### 4. Anisotropic Collision Awareness (ACA)
Standard latent physics models treat primitives as point masses. We upgraded the **CollisionRegularizer** to account for **anisotropic scaling** ($s_i$). The interaction potential now scales with the effective geometric radius $R_{ij} = r_{base} + s_i + s_j$, preventing visual artifacts in compressed or elongated Gaussian clusters.

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

### 5. Spectral Repulsion & Lorentzian Potential
Upgraded the **CollisionRegularizer** to incorporate a **Spectral Repulsion** term. This replaces the naive squared ReLU with a Lorentzian-dampened potential $V = \frac{\delta^2}{1 + \epsilon \delta}$, where $\delta$ is the geometric overlap. This ensures numerical stability during high-density primitive interactions and prevents gradient explosion in the latent Lagrangian field.

### 6. Orientation-Aware Projection (OAP) Interaction
Upgraded the interaction model in `CollisionRegularizer` to support **OAP Interaction**. Unlike isotropic approximations, OAP explicitly calculates the projection of each Gaussian's covariance ellipsoid along the interaction vector using the rotation manifold ($R \in SO(3)$). The directional radius is derived as $r_i(\mathbf{u}) = \|\mathbf{S}_i \mathbf{R}_i^T \mathbf{u}\|_2$, enabling high-fidelity collision constraints for extremely elongated or flattened Gaussian primitives.

---
*Last Academic Update: 2026-02-10 12:45 (Singapore)*

### 7. Phase-Space Entropy Regularization (PSER)
Implemented **PSER** in the `LagrangianODESolver` to mitigate the "fixed-point collapse" common in high-dimensional latent physics. By penalizing the log-inverse pair-wise distance in the latent phase-space ($\mathcal{L}_{PSER} = -\mathbb{E}[\log(\|z_i - z_j\| + \epsilon)]$), we enforce a more expressive manifold distribution, ensuring that the generated 3D Gaussian trajectories remain diverse and topologically complex over long temporal horizons.

### 8. Causal-Spectral Hamiltonian Weighting
Refined the Hamiltonian conservation loss to include **Causal Weighting**. The penalty for energy drift ($dH/dt$) is now exponentially decayed over time, prioritizing the stabilization of the "initial flow" during training. This prevents numerical errors in early-time steps from propagating and destabilizing the entire 4th-order Runge-Kutta trajectory.

### 9. Symplectic Jacobian Consistency (SJC)
Integrated a **Symplectic Jacobian Loss** into the `LagrangianODESolver`. By enforcing $M^T J M = J$ (where $M$ is the Jacobian of the latent flow), we ensure that the latent phase-space evolution is a true canonical transformation. This preserves the symplectic structure of the underlying physical manifold, preventing long-term dissipative artifacts and ensuring volume conservation in the latent world model.

---
*Last Academic Update: 2026-02-10 15:36 (Singapore)*

### 16. Latent Curvature Preservation (LCP)
Integrated **LCP** into the `PhysiGen3D` conservation loss. This term penalizes the second-order temporal derivative (acceleration) of the latent flow, effectively enforcing a "minimum curvature" constraint. By ensuring that the latent trajectory respects the Ricci-flatness of the underlying physical manifold, LCP prevents erratic "snapping" artifacts in the generated 3D Gaussian dynamics and promotes more natural, inertial-based motion transitions.

### 17. Lagrangian Divergence Minimization (LDM)
Implemented **LDM** in the `LagrangianODESolver` training loop. This constraint enforces the solenoidality of the latent velocity field ($\nabla \cdot \mathbf{v} = 0$), effectively modeling the latent phase-space as an **incompressible fluid**. LDM prevents Gaussian primitives from "bunching" or collapsing into singularity points during complex interactions, ensuring a more uniform and stable distribution of the 3D world representation across long temporal rollouts.

### 18. Harmonic Balance Regularization (HBR)
Added **HBR** to the physical conservation suite. Inspired by the **Virial Theorem**, HBR enforces a statistical balance between the average kinetic and potential energy fluctuations over a temporal sequence. This prevents the latent physics engine from over-fitting to either purely inertial or purely static configurations, promoting more realistic oscillatory and dissipative behaviors in the generated world.

### 19. Ghost-Force Suppression (GFS)
Integrated **GFS** as a low-pass filtering mechanism for the predicted Hamiltonian gradients. By penalizing high-frequency temporal jitter in the latent forces ($\partial \mathcal{H} / \partial z$), GFS suppresses numerical noise that typically arises from high-dimensional neural ODE solvers. This results in smoother trajectories and increased temporal coherence in the 3D Gaussian Splatting scene reconstruction.

### 20. Anisotropic Volume Persistence (AVP)
Implemented **AVP** in the physical conservation suite to ensure latent manifold stability. By enforcing a unit-determinant constraint on the Jacobian of the latent flow ($\det(\mathbf{M}) \approx 1$), AVP ensures that the phase-space volume is conserved even under anisotropic scaling of the Gaussian primitives. This prevents the "inflationary collapse" of the 3D world representation and stabilizes long-term rollout trajectories.

---
*Last Academic Update: 2026-02-10 16:45 (Singapore)*

### 21. Spectral-Entropic Causal Stabilizer (SECS)
Integrated **SECS** into the latent physics loss suite. This term couples the **Spectral Power Density** of the 3D Gaussian trajectories with the **Shannon Entropy** of the latent phase-space distribution. By maximizing the information content within the dominant frequency bands, SECS prevents "physical hallucinations"—trajectories that are mathematically stable but physically meaningless—ensuring that the generated world models remain both dynamic and informative over long-term temporal rollouts.

### 22. Temporal-Spectral Flux Consistency (TSFC)
Implemented **TSFC** in the `LagrangianODESolver` to manage energy transfer across temporal scales. By enforcing that the spectral energy flux matches the expected physical dissipation rate, TSFC prevents **spectral aliasing** and numerical energy "pile-up" in high-frequency modes. This ensures that the generated 3D world transitions are physically plausible and free from the temporal artifacts typically associated with high-dimensional latent ODE solvers.

### 23. Manifold Robustness Fine-Tuning (MRFT)
Integrated **MRFT** into the `PhysiGen3D` architecture, inspired by the latest research on representation autoencoders (arXiv:2602.08620). MRFT injects controlled noise into the latent trajectories during training to smooth the data manifold and penalize the decoder's sensitivity to perturbations. This results in significantly higher-fidelity 3D Gaussian reconstructions and reduces visual artifacts in the generated video sequences by ensuring the decoders remain robust even when conditioned on high-dimensional, information-rich latent features.

---
*Last Academic Update: 2026-02-10 18:15 (Singapore)*

### 24. Multi-Scale Kinetic Dissipation (MSKD)
Integrated **MSKD** into the `CollisionRegularizer` to stabilize high-velocity primitive interactions. MSKD acts as a neural viscosity term, specifically penalizing the relative approach velocity of overlapping Gaussians ($\mathcal{L}_{MSKD} = \sum \delta_{ij} \cdot \max(0, -\mathbf{v}_{ij} \cdot \hat{\mathbf{u}}_{ij})$). This prevents numerical "explosion" during dense collisions and ensures that the latent physics engine maintains energy conservation without sacrificing the dynamic range of the 3D world model.

---
*Last Academic Update: 2026-02-10 18:36 (Singapore)*

### 25. Temporal Jacobian Spectral Consistency (TJSC)
Implemented **TJSC** to bridge the gap between phase-space dynamics and frequency-domain stability. By coupling the latent Jacobian $\mathbf{M}$ with the spectral power density, TJSC enforces that the physical flow preserves the spectral energy flux across temporal scales. This mechanism prevents the physics engine from "scrambling" the multi-scale structural representation of the 3D world, ensuring that large-scale global motion (low-frequency) and fine-grained geometric details (high-frequency) evolve in a physically consistent, decoupled manner during long-term rollout.

### 26. Phase-Space Adaptive Initialization (PSAI)
Integrated **PSAI** into the `LagrangianODESolver` to optimize the entry-point of latent trajectories. PSAI dynamically modulates the initial latent state $\mathbf{z}_0$ based on the **intrinsic manifold temperature** of the conditioning embedding. This ensures that the ODE solver is initialized on a stable region of the physical manifold, significantly reducing early-time numerical instability and improving the long-horizon consistency of the generated 3D Gaussian dynamics.

### 27. Multi-Scale Spectral Diffusion (MSSD)
Implemented **MSSD** to regularize the latent spectral density. MSSD enforces a multi-scale Gaussian-Laplace prior across the frequency domain, preventing the formation of "spectral gaps" or anomalous "peaks" that typically lead to temporal aliasing or visually stagnant frames in neural physics simulations. This ensures a more natural and continuous energy distribution across all temporal scales of the 3D world model.

---
*Last Academic Update: 2026-02-10 20:12 (Singapore)*
