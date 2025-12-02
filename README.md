# 🤖 Anomaly Detection in Robotic Manipulator Data (RoAD)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python" alt="Python Version"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/AutoGluon-051c2c?style=for-the-badge&logo=dataiku&logoColor=white" alt="AutoGluon"/>
  <img src="https://img.shields.io/badge/Anomaly%20Detection-Research-9300ff?style=for-the-badge" alt="Anomaly Detection"/>
</p>

This project implements and compares different methodologies (Deep Learning with PyTorch and Automated Machine Learning with AutoGluon) for **anomaly detection** in multivariate time series data from a robotic environment, utilizing the **RoAD (Robot Anomaly Detection)** dataset.

---

## 📝 Table of Contents

1.  [🚀 Project Overview](#-project-overview)
2.  [⚙️ RoAD Dataset](#️-road-dataset)
    * [Data Structure](#data-structure)
    * [Dataset Split and Scenarios](#dataset-split-and-scenarios)
3.  [Anomaly Detection Models](#anomaly-detection-models)
    * [Deep Learning Models (PyTorch)](#deep-learning-models-pytorch)
    * [AutoGluon Baseline (Classical ML/Ensemble)](#autogluon-baseline-classical-mlensemble)
4.  [Data Processing Pipeline](#data-processing-pipeline)
5.  [📊 Results and Comparison](#-results-and-comparison)
    * [Test Metrics Summary](#test-metrics-summary)
    * [Visualizations](#visualizations)

---

## 🚀 Project Overview

The goal is to classify temporal windows of robotic data as either **Normal (0)** or **Anomalous (1)**. Two main approaches are compared for this binary classification challenge on time series:

1.  **Deep Learning (PyTorch)**: Custom models (1D CNN and MLP) that process raw time windows.
2.  **Automated Machine Learning (AutoGluon)**: An approach based on extracting key statistics (mean, standard deviation, min, max) from the window, converting the problem into a tabular classification task.

The project rigorously prevents **Data Leakage** by performing Min-Max normalization exclusively on the training data statistics and strictly separating records before window creation.

---

## ⚙️ RoAD Dataset

The project uses the **RoAD Dataset** (available on [GitLab](https://gitlab.com/AlessioMascolini/roaddataset)), a collection of time series describing the behavior of a robotic manipulator in normal and anomalous scenarios.

### Data Structure

Each temporal record contains **86 numerical features** relating to:

* **Global Metrics:** Apparent power, current, voltage, frequency, power factor, etc.
* **Joints (1-7):** Acceleration (x, y, z), gyroscope (x, y, z), quaternions (qw, qx, qy, qz), and temperature for each of the 7 joints.

**Window Labeling:** A window is labeled **Anomalous (1)** if *any* timestep within that window contains an anomaly label.

### Dataset Split and Scenarios

The raw data is divided into four main scenarios, used to structure the training and test sets:

| Scenario | Description | Usage in Windows |
| :--- | :--- | :--- |
| **`training` & `control`** | Normal robot operations. | Primary source for the **Training Set** (labeled `0`). |
| **`collision`** | Anomalies due to impacts. | Used to populate the **Test Set** (labeled `1`). |
| **`weight`** | Anomalies due to excessive load. | Used to populate the **Test Set** (labeled `1`). |
| **`velocity`** | Anomalies due to abnormal speed/acceleration. | Used to populate the **Test Set** (labeled `1`). |

---

## Anomaly Detection Models

### Deep Learning Models (PyTorch)

PyTorch models (`main.py`) operate directly on normalized time windows of shape $(C, T)$, where $C=86$ and $T=256$.

#### 1. CNN 1D (`models/cnn_1d.py`)


The **RoADCNN1D** model is a deep convolutional network designed for time series analysis.

* **Architecture:** It uses 4 stages of **Residual Blocks** with 1D convolutions.
* **Dilation:** It employs increasing temporal dilation (1, 2, 4, 8) to expand the receptive field and capture long-range temporal dependencies without changing the temporal resolution.
* **Output Head:** A $1\times1$ convolution (`frame_head`) produces logits per timestep, which are then aggregated using **Top-K Pooling** (default) to yield the final sequence logit (`seq_logit`).

#### 2. MLP Classifier (`models/mlp.py`)

* **Architecture:** Based on the **WindowedMLP** (an Autoencoder structure), adapted for classification.
* **Operation:** The input window $(B, C, T)$ is flattened, and the **Encoder** part of the MLP produces a latent vector **$z$** (bottleneck, dimension 32 by default). A dedicated linear layer (Classification Head) then projects $z$ to the final classification logit.

---

### AutoGluon Baseline (Classical ML/Ensemble)

The AutoGluon approach (`autogluon.py`) transforms the sequential problem into a standard tabular one.

* **Feature Engineering:** For each of the 86 features within the $T=256$ window, four statistics are extracted: **mean, standard deviation, minimum, and maximum**. This results in a feature vector of $86 \times 4 = 344$ dimensions per window.
* **Modeling:** AutoGluon automatically trains and ensembles various Machine Learning models (e.g., Random Forests, LightGBM, Neural Networks) to find the best classifier.
* **Test Stride:** A **stride equal to `window_size` (256)** is used in the test set to ensure windows do not overlap, providing an unbiased performance estimate.

---

## Data Processing Pipeline

The pipeline is consistently applied across all scripts for fair comparison:

1.  **Normalization:** Global **Min-Max Scaling** statistics are computed **only** on the raw records from the training set (`training` + `control`).
2.  **Transformation:** The calculated Min-Max Scaling is applied to *all records* (train and test).
3.  **Record-Level Split:** Normalized records are split 70/30 into train/test subsets for each scenario.
4.  **Windowing:** Temporal windows of size **$T=256$** are created:
    * **Train Windows:** Stride of **128** (overlap for data augmentation).
    * **Test Windows:** Stride of **256** (no overlap, for honest evaluation).

---

## 📊 Results and Comparison

All results are saved in the `runs_pytorch` and `runs_autogluon` directories, following the execution of `main.py` and `autogluon.py`. The `plots.py` and `plots_autogluon.py` scripts are used to generate visual summaries.

### Test Metrics Summary

| Model | ROC AUC (Test) | Best F1 (Test) | Accuracy @ 0.5 | BCE Loss (Test) |
| :--- | :--- | :--- | :--- | :--- |
| **CNN\_1D (PyTorch)** | [] | [] | [] | [] |
| **MLP\_Classifier (PyTorch)** | [] | [] | [] | [] |
| **Best Model (AutoGluon)** | [] | [] | [] | [] |

### Visualizations

The output directories contain the following plots for detailed analysis:

* **`comparison_*.png`**: Bar charts comparing models across key metrics (ROC AUC, Best F1, Accuracy @ 0.5).
* **`detailed_metrics_vs_threshold.png`**: Learning curve showing F1 Score, Accuracy Class 0 (Normal), and Accuracy Class 1 (Anomaly) as a function of the classification threshold.
* **`confusion_matrix_norm.png`**: Normalized Confusion Matrix calculated at the **Best F1 Threshold**.
* **`feature_importance.csv`**: (AutoGluon only) List of the most important features (statistics) used by the best models.