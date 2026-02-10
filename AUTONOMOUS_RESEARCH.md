# PhysiGen-WM Autonomous Research Log

## 2026-02-10 12:45
- **Step**: Refined `PhysiGen3D` model with Phase-Space Entropy Regularization (PSER).
- **Reasoning**: Latent trajectories in world models often collapse to static states. PSER encourages movement on the manifold.
- **Changes**: 
    - Updated `src/models/physigen.py`: Added `entropy_reg` to `calculate_conservation_loss`.
    - Updated `README.md`: Documented PSER and Causal-Spectral weighting.
- **Status**: Stable. Ready for next training epoch.
