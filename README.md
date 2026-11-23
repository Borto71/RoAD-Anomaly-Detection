<h1 align="center">Anomaly Detection on the RoAD Dataset</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/AutoGluon-Tabular-0B3D91?style=flat-square" alt="AutoGluon"/>
</p>

<p align="center">
  Binary anomaly detection on multivariate time series from a robotic manipulator,<br/>
  comparing deep learning models (1D CNN, MLP) against an AutoGluon tabular baseline.
</p>

---

## Overview

Each sample is a window of **256 timesteps × 86 features** (power metrics, IMU and temperature for 7 joints) from the [RoAD dataset](https://gitlab.com/AlessioMascolini/roaddataset). A window is labeled **anomalous** if any of its timesteps is anomalous.

| Approach | Input | Description |
| :-- | :-- | :-- |
| **CNN 1D** | Raw window `(86, 256)` | Dilated residual blocks with top-k temporal pooling |
| **MLP** | Flattened window | Encoder to a 32-d latent space, linear classification head |
| **AutoGluon** | 344 statistics | Mean, std, min, max per feature; automated ensembling |

Min-max normalization is fitted on training records only, and the train/test split is performed before windowing to prevent data leakage.

## Installation

```bash
git clone https://github.com/Borto71/RoAD-Anomaly-Detection.git
cd RoAD-Anomaly-Detection

python -m venv .venv && source .venv/bin/activate
pip install torch numpy pandas scikit-learn matplotlib seaborn autogluon.tabular
```

The `RoADDataset` loader must be importable; obtain it from the [official RoAD repository](https://gitlab.com/AlessioMascolini/roaddataset).

## Usage

```bash
python main.py        # train and evaluate the PyTorch models  -> runs_pytorch/
python autogluon.py   # train and evaluate the AutoGluon model -> runs_autogluon/
```

Plots are generated with `utils/plots.py` and `utils/plots_autogluon.py`.

## Results

Test set, non-overlapping windows (stride 256).

| Model | ROC AUC | Best F1 | Accuracy @ 0.5 |
| :-- | :--: | :--: | :--: |
| **CNN 1D** | **0.73** | **0.67** | 0.59 |
| **MLP** | 0.65 | 0.64 | **0.70** |
| **AutoGluon** | — | — | — |

The CNN 1D achieves the best ranking quality (ROC AUC) and F1 score, while the MLP yields higher accuracy at the default threshold. AutoGluon results are pending.

## Project Structure

```
├── main.py              # PyTorch training and evaluation
├── autogluon.py         # AutoGluon baseline
├── models/              # CNN 1D and MLP architectures
├── data/                # Data utilities
└── utils/               # Early stopping and plotting
```
