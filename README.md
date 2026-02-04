# Cross-Domain Unsupervised Crack & Anomaly Detection

A high-performance anomaly detection pipeline using self-supervised Vision Transformers (DINO) for industrial quality control on the MVTec dataset.

## 🚀 Accuracy Overview
*   **Overall Normalized AUROC:** 91.16%
*   **Top Categories:** Bottle (1.0), Tile (0.99), Leather (0.99), Cable (0.99)
*   **Hard Categories:** Grid (0.78), Screw (0.67)

## 🧠 How It Works

### 1. Hybrid Feature Extraction (`feature_extraction.py`)
The system uses a `vit_small_patch16_224.dino` backbone. It extracts:
*   **Global Context:** CLS token from the final layer.
*   **Local Detail:** Concatenated patch tokens from Layer 11 and Layer 12.
*   **High-Res Mapping:** The 14x14 grid is interpolated to 28x28 (784 tokens) to detect micro-cracks.

### 2. Memory Bank & Coreset (`memory_bank.py`)
Instead of storing every normal patch, we use **MiniBatchKMeans** to select 5,000 representative "prototypes" (Coreset) per category. This ensures fast inference while maintaining a diverse baseline of "normal" behavior.

### 3. Adaptive Anomaly Scoring (`anomaly_scoring.py`)
The anomaly score is a weighted combination of:
*   **Global Score:** Distance between the test image's CLS token and the nearest normal CLS token.
*   **Patch Score:** Mean distance of the Top-K most anomalous patches. 
    *   **Textures:** Uses Top 15% (captures surface variance).
    *   **Objects:** Uses Top 6% (captures localized defects).

### 4. Explainable AI (`visualize_anomalies.py`)
Generates JET-color heatmaps overlaid on the original image. It also extracts the most anomalous patch to show exactly what region triggered the alarm.

## 🛠 Project Strengths
*   **Domain Robustness:** Maintains >98% accuracy even with ±50% brightness shifts or Gaussian blur.
*   **Zero-Shot Transfer:** A bank trained on `metal_nut` can detect defects in `screw` (semantic generalization).
*   **Unsupervised:** Requires zero anomalous examples for training; learns purely from "good" data.

## 📂 Directory Structure
*   `embeddings/`: Saved raw tensors for training/testing.
*   `memory_bank/`: Category-specific PCA models and Coreset centroids.
*   `outputs/`: Heatmaps, XAI figures, and evaluation reports.
*   `config.py`: Centralized hyperparameters.
