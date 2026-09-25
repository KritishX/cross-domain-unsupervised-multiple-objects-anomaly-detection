# Cross-Domain Unsupervised Anomaly Detection

An image-based industrial inspection project that learns **normal** visual patterns and flags unusual regions without needing labelled defect examples for training. It uses a self-supervised DINO Vision Transformer, nearest-neighbour memory banks, and spatial anomaly maps to inspect the 15 MVTec AD object and texture categories.

> This repository contains the pipeline code. The MVTec AD dataset, generated embeddings, and trained memory banks are intentionally excluded from version control because of their size and dataset licence.

## What it does

- Extracts a global CLS descriptor and an upsampled 28 x 28 grid of local DINO features.
- Builds a per-category normal reference bank from training images, compressed with MiniBatchKMeans and PCA.
- Scores a test image using a weighted combination of global and local nearest-neighbour distances.
- Uses adaptive Top-K aggregation: 15% of patches for textures and 6% for discrete objects.
- Produces CSV metrics, cross-domain transfer results, robustness tests, heatmap overlays, and a Streamlit inspection interface.

## Pipeline

```text
normal training images
        |
        v
DINO ViT feature extraction --> saved embeddings
        |
        v
PCA + normal memory bank --------> test-image scoring
                                         |
                 +-----------------------+-----------------------+
                 v                       v                       v
             AUROC report            heatmaps             Streamlit UI
```

## Quick start

The project has been developed as standalone Python scripts. Use Python 3.9+ and install the runtime packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision timm numpy pandas scikit-learn joblib tqdm pillow matplotlib opencv-python streamlit
```

Download the [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) and place its category folders under `dataset/`. Keep the dataset licence file with the data.

Run the core pipeline from the repository root:

```bash
# 1. Extract DINO features for one or more categories
python feature_extraction.py --target bottle cable

# 2. Build normal memory banks
python memory_bank.py --target bottle cable

# 3. Score the test images and write outputs/anomaly_scores.csv
python anomaly_scoring.py

# 4. Calculate AUROC, F1, and category-specific thresholds
python evaluate.py
```

Optional analyses:

```bash
python cross_domain_eval.py    # evaluates configured source-to-target transfers
python robustness_test.py      # brightness and blur robustness for selected categories
python visualize_anomalies.py  # creates heatmap and explainability images
streamlit run gui_app.py       # opens the interactive inspector
```

The first DINO model creation may download pretrained weights through `timm`. The scripts select CUDA when it is available, otherwise they run on CPU.

## Repository guide

| File | Purpose |
| --- | --- |
| `feature_extraction.py` | DINO feature extraction and 28 x 28 patch-grid upsampling |
| `memory_bank.py` | Normal-feature coreset construction and PCA fitting |
| `anomaly_scoring.py` | Global/local nearest-neighbour anomaly scoring |
| `evaluate.py` | Per-category AUROC, F1, and threshold report |
| `cross_domain_eval.py` | Configured cross-category transfer evaluation |
| `robustness_test.py` | Brightness and Gaussian-blur stress tests |
| `visualize_anomalies.py` | Heatmap overlays and anomalous-patch views |
| `gui_app.py` | Streamlit interface for inspecting uploaded images |
| `config.py` | Shared category metadata and score settings |

## Results snapshot

The following values were documented with the original project artifacts and are retained as a historical snapshot, not a fresh reproducibility claim:

- Overall normalized AUROC: **91.16%**
- Strongest recorded categories: bottle, tile, leather, and cable
- More challenging recorded categories: grid and screw

Re-run the pipeline with your environment, dataset copy, and dependency versions before comparing or reporting results.

## Data and licence

MVTec AD is attributed to Bergmann et al., *A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection* (CVPR 2019). The bundled dataset notice identifies it as **CC BY-NC-SA 4.0**; review that licence and the upstream dataset terms before redistributing data or using it commercially. This repository does not add a separate licence for the project code—add one before treating the code as reusable under a particular set of terms.

## Limitations

- The implementation is category-specific: a memory bank must exist for the selected inspection domain.
- Thresholds are derived from evaluation data and should be recalibrated for a real production line.
- The supplied experiments are research-oriented and do not establish production safety, latency, or robustness guarantees.
