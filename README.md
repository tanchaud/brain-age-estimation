# Brain Age Estimation from MRI

A Python pipeline for predicting brain age from T1-weighted MRI scans using deep learning.

## Overview

This project implements brain age prediction from T1-weighted MRI scans. It supports two modelling approaches:

- **3D ResNet-18** *(current, recommended)*: End-to-end 3D CNN trained directly for age regression — no separate feature extraction step
- **VGG16 + SVR** *(legacy)*: VGG16 feature extraction from slice triplets combined with Support Vector Regression using Early or Late Fusion

## Features

- End-to-end 3D ResNet-18 trained on full MRI volumes
- MONAI-based preprocessing pipeline (reorientation, resampling, normalisation)
- Dataset cleaning step to automatically detect and skip corrupt/blank scans
- Mixed-precision training (AMP) for fast Colab GPU execution
- Two interactive Streamlit web demos (3D ResNet and VGG16+SVR)
- Comprehensive evaluation metrics (MAE, RMSE, Pearson correlation)
- Memory-efficient design supporting 500+ subjects

## Project Structure

```
.
├── app_resnet3d.py              # Streamlit demo — 3D ResNet-18 (current)
├── app.py                       # Streamlit demo — VGG16 + SVR (legacy)
├── BrainAge_3DResNet_Colab.ipynb  # Colab notebook — end-to-end 3D ResNet training
├── brain_age_project_main.py    # Legacy workflow script (VGG16 + SVR)
├── feature_extraction.py        # VGG16 triplet feature extraction
├── early_fusion.py              # Early fusion SVR
├── late_fusion.py               # Late fusion with weak learners
├── concat_predictions.py        # Prediction combination strategies
├── data_split.py                # Train/test splitting (age-sorted, 75/25)
├── load_nifti.py                # NIfTI file loading
├── mae.py                       # Mean Absolute Error metric
├── rmse.py                      # Root Mean Squared Error metric
├── pearson_corr.py              # Pearson correlation metric
└── sanity_check.py              # Dependency verification
```

## Requirements

### 3D ResNet-18 (current model)

- Python 3.8+
- PyTorch 2.x
- MONAI
- nibabel
- matplotlib, pandas, openpyxl

```bash
pip install torch monai nibabel matplotlib pandas openpyxl streamlit
```

### VGG16 + SVR (legacy)

```bash
pip install tensorflow numpy pandas scikit-learn opencv-python nibabel matplotlib openpyxl
```

## Dataset

Designed for the [IXI Dataset](https://brain-development.org/ixi-dataset/):
- T1-weighted MRI scans (NIfTI format)
- Demographics Excel file with subject ages

Expected structure:
```
Data_IXI/
├── IXI-T1/
│   ├── IXI002-Guys-0828-T1.nii.gz
│   ├── IXI012-HH-1234-T1.nii.gz
│   └── ...
└── IXI.xls   (columns: IXI_ID, AGE)
```

## Usage

### 3D ResNet-18 — Google Colab (Recommended)

1. Upload data to Google Drive
2. Open `BrainAge_3DResNet_Colab.ipynb` in [Google Colab](https://colab.research.google.com)
3. Enable GPU: *Runtime → Change runtime type → T4 GPU*
4. Edit the three path variables (`NII_DIR`, `DEMOGRAPHICS`, `CHECKPOINT_DIR`)
5. Run all cells — best model saved automatically to `CHECKPOINT_DIR/best_resnet3d.pth`

### Streamlit Demo — 3D ResNet-18

```bash
streamlit run app_resnet3d.py
```

1. Download `best_resnet3d.pth` from Google Drive to your local machine
2. Enter its local path in the **Model Checkpoint** sidebar field
3. Upload any `.nii` / `.nii.gz` file to get a brain age prediction

### Streamlit Demo — VGG16 + SVR (legacy)

```bash
streamlit run app.py
```

## Methods

### 3D ResNet-18 (Current)

| Step | Detail |
|------|--------|
| Preprocessing | RAS reorientation → 2 mm isotropic resampling → pad/crop to 96³ → z-score normalisation |
| Architecture | 3D ResNet-18: stem (7³ conv) + 4 residual stages (64→128→256→512 ch) + AdaptiveAvgPool → FC(1) |
| Training | L1 loss, Adam (lr=1e-4), cosine LR annealing, dropout=0.3, AMP, batch size 2 |
| Dataset cleaning | Corrupt files, <3D volumes, dimensions <32 voxels, NaN/Inf, blank scans are dropped automatically |

### VGG16 + SVR (Legacy)

- MRI volumes divided into non-overlapping triplets of 3 consecutive axial slices
- Each triplet resized to 224×224 and fed to VGG16 (`block5_pool`, 25,088-dim features)
- **Early Fusion**: mean-aggregate features → single linear SVR (C=98, ε=0.064)
- **Late Fusion**: one SVR per triplet position → combine predictions by weighted mean / oracle

## Results

### 3D ResNet-18 (End-to-End)

| Metric | Value |
|--------|-------|
| MAE | **~4–5 years** |
| RMSE | **~6–7 years** |
| Pearson r | **~0.90–0.95** |

### VGG16 + SVR (Legacy)

| Method | MAE (years) | RMSE (years) | Pearson r |
|--------|-------------|--------------|-----------|
| Early Fusion | ~7–8 | ~12–14 | ~0.75–0.80 |
| Late Fusion  | ~7–9 | ~12–15 | ~0.70–0.78 |

*3D ResNet-18 results reflect end-to-end learning on the full 3D volume vs. the legacy approach's 2D slice-based feature extraction.*

## Output Files

| File | Description |
|------|-------------|
| `best_resnet3d.pth` | Best 3D ResNet-18 checkpoint (saved to Colab `CHECKPOINT_DIR`) |
| `brain_age_predictions_resnet3d.csv` | Per-subject predictions + Brain Age Gap |
| `training_curves.png` | Loss / MAE / RMSE / Pearson r over epochs |
| `predicted_vs_true_age.png` | Scatter plot of predictions vs. true ages |

## Citation

If you use this code, please cite:

```bibtex
@software{brain_age_estimation,
  title  = {Brain Age Estimation from MRI},
  author = {Tanchaud},
  year   = {2024},
  url    = {https://github.com/tanchaud/brain-age-estimation}
}
```

## License

This project is available for academic and research purposes.

## Acknowledgments

- [IXI Dataset](https://brain-development.org/ixi-dataset/) for providing the brain MRI data
- [MONAI](https://monai.io/) for medical imaging transforms and utilities
- [PyTorch](https://pytorch.org/) for the deep learning framework
