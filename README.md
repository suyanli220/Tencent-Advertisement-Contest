# TAAC 2026: DeepContextNet Baseline

![Competition](https://img.shields.io/badge/Contest-Tencent--Advertisement--2026-blue)
![Python](https://img.shields.io/badge/python-3.9+-green.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)
![Model](https://img.shields.io/badge/Architecture-HSTU--Muon-orange)

This repository contains the high-performance baseline for the **2026 Tencent Advertisement Algorithm Contest (TAAC)**. The solution features **DeepContextNet**, a sophisticated sequence-based architecture optimized with the **Muon** second-order approximation optimizer.

---

## 🚀 Highlights

- **HSTU Core:** Implements the *Hierarchical Sequential Transformer Unit* for linear-complexity sequence modeling.
- **Muon Optimizer:** Utilizes `MatrixUnitaryOptimizer` for gradient orthogonalization, leading to faster convergence and better generalization in sparse feature spaces.
- **Global CLS Integration:** Aggregates static user profiles, item features, and dynamic behavior sequences into a single global context token.
- **Time-Aware Decay:** Captures temporal dynamics using log-scale time bucketing to model the recency of user interactions.
- **Mixed Precision Ready:** Fully compatible with `torch.cuda.amp` for memory-efficient training on modern GPUs.

## 🧠 Model Architecture

DeepContextNet follows a four-stage execution logic:

1.  **Unified Embedding Layer:** Maps sparse categorical IDs (`int_array`) and dense numerical vectors (`float_array`) into a shared $d$-dimensional latent space.
2.  **Sequential Synthesis:** Merges `Item_ID`, `Action_Type`, and `Temporal_Bucket` embeddings via additive fusion.
3.  **Deep Interaction:** Passes the concatenated `[CLS, User, Item, Sequence]` tokens through multiple **HSTU blocks** with Rotary Positional Bias.
4.  **Projection Head:** A non-linear bottleneck layer translates the global context into a final CTR (Click-Through Rate) probability.

[Image of Transformer model architecture for recommendation systems]

---

## 🛠️ Installation & Usage

### Prerequisites
- Python 3.9+
- PyTorch 2.0+
- NVIDIA GPU with CUDA 11.8+ (Recommended)

## 🚀 Quick Start

### 1. Data Placement
Ensure your dataset is placed in the `data/` directory. The preprocessing script expects the following path:
`data/sample_data.parquet`

### 2. Training
Launch the training pipeline using the default configuration:

python main.py

### ⚙️ Configuration
Adjust hyperparameters within the `RuntimeParams` class in `main.py` to tune the model for specific hardware or dataset scales:

* **HIDDEN_DIM**: Latent space dimensionality. Controls the capacity of the embeddings and Transformer blocks (Default: `128`).
* **BLOCK_DEPTH**: Number of stacked Transformer layers. Increasing this allows for higher-order feature interactions (Default: `4`).
* **LEARNING_RATE**: Optimized for Muon performance. Recommended starting point is `2e-4`.
* **SESSION_GAP**: Temporal threshold (in seconds) used to partition interaction sequences into distinct user sessions (Default: `1800s`).

---

### 📈 Muon Optimization Strategy
Unlike standard first-order optimizers (such as AdamW), the **Muon** optimizer implemented in this repository utilizes **SVD-based orthogonalization** on the parameter update matrices. 



By applying this technique, the training process benefits from:
1.  **Directional Learning**: It decouples the learning of feature directions from their magnitudes, allowing for more precise updates in high-dimensional spaces.
2.  **Gradient Stability**: Effectively mitigates the "vanishing gradient" problem often encountered in deep CTR (Click-Through Rate) models.
3.  **Embedding Efficiency**: Provides a remarkably stable training trajectory, especially for high-cardinality embedding layers that are prone to noise.

---

### 📜 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---
*Good luck with the TAAC 2026 competition! If this baseline helps your ranking, please consider giving the repository a ⭐.*
