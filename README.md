# PhysiGen-WM: Physics-Informed World Model for 3D Scenes

PhysiGen-WM is a generative world model that integrates **Physics-Informed Neural Networks (PINNs)** with **3D Gaussian Splatting** for physically consistent scene evolution.

## 🚀 Core Features
- **Physics-Grounded Generation**: Latent dynamics governed by learned ODEs and physical conservation laws.
- **3D Gaussian Splatting**: High-fidelity differentiable rendering for spatial consistency.
- **Spectral-Causal Stabilization**: Novel loss functions (SCA, SCEC) to prevent long-horizon drift and temporal jitter.

## 🛠 Installation
```bash
git clone https://github.com/Chenypovo/PhysiGen-WM.git
cd PhysiGen-WM
pip install -r requirements.txt
```

## 📈 Training
To train with synthetic physics data:
```bash
python3 scripts/train_full.py
```

## 🔮 Demo Generation
Generate a 3D generative world preview:
```bash
python3 scripts/generate_generative_demo.py
```

---
*Developed by Yipeng Chen for PhD Portfolio research in AI & World Models.*
