# Brain Age Estimation from MRI

A Python pipeline for predicting brain age from T1-weighted MRI scans using deep learning features and machine learning regression.

## Overview

This project implements brain age prediction using VGG16 features extracted from MRI brain scans. It compares two fusion strategies:

- **Early Fusion**: Aggregate features across brain slices, then train a single SVR model
- **Late Fusion**: Train separate SVR models for each slice position, then combine predictions

## Features

- Memory-efficient processing (handles 500+ subjects without running out of memory)
- VGG16-based deep feature extraction from MRI triplet slices
- Support Vector Regression (SVR) for age prediction
- Both Early and Late Fusion strategies
- Google Colab notebook with GPU acceleration
- Comprehensive evaluation metrics (MAE, RMSE, Pearson correlation)

## Project Structure

```
.
├── brain_age_project_main.py    # Main workflow script
├── Brain_Age_Colab.ipynb        # Google Colab notebook (GPU-accelerated)
├── feature_extraction.py        # CNN feature extraction utilities
├── early_fusion.py              # Early fusion implementation
├── late_fusion.py               # Late fusion with weak learners
├── data_split.py                # Train/test splitting
├── load_nifti.py                # NIFTI file loading
├── concat_predictions.py        # Prediction combination strategies
├── mae.py                       # Mean Absolute Error metric
├── rmse.py                      # Root Mean Squared Error metric
├── pearson_corr.py              # Pearson correlation metric
└── sanity_check.py              # Dependency verification
```

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- scikit-learn
- OpenCV (cv2)
- nibabel
- matplotlib
- openpyxl / xlrd (for Excel file support)

Install dependencies:
```bash
pip install tensorflow numpy pandas scikit-learn opencv-python nibabel matplotlib openpyxl xlrd
```

Or with conda:
```bash
conda install -c conda-forge tensorflow numpy pandas scikit-learn opencv nibabel matplotlib openpyxl xlrd
```

## Dataset

This pipeline is designed for the [IXI Dataset](https://brain-development.org/ixi-dataset/):
- T1-weighted MRI scans (NIFTI format)
- Demographics file with subject ages (Excel format)

Expected data structure:
```
Data_IXI/
├── IXI-T1/
│   ├── IXI002-Guys-0828-T1.nii.gz
│   ├── IXI012-HH-1234-T1.nii.gz
│   └── ...
└── IXI.xls  (demographics with IXI_ID and AGE columns)
```

## Usage

### Local Execution

1. Update the data paths in `brain_age_project_main.py`:
```python
DATA_BASE_DIR = '/path/to/your/Data_IXI'
```

2. Run the workflow:
```bash
python brain_age_project_main.py
```

### Google Colab (Recommended for faster processing)

1. Upload your data to Google Drive
2. Open `Brain_Age_Colab.ipynb` in [Google Colab](https://colab.research.google.com)
3. Enable GPU: Runtime > Change runtime type > GPU
4. Update the data paths and run all cells

## Methods

### Feature Extraction

- MRI volumes are divided into non-overlapping triplets of consecutive slices
- Each triplet is treated as a pseudo-RGB image and resized to 224x224
- VGG16 (pretrained on ImageNet) extracts features from `block5_pool` layer
- Output: 25,088-dimensional feature vector per triplet

### Early Fusion

1. Extract features for all triplets per subject
2. Aggregate by taking the mean across triplets
3. Train a single linear SVR on aggregated features
4. Predict age for test subjects

### Late Fusion

1. For each triplet position (e.g., 40 positions), train a separate SVR
2. Each "weak learner" predicts age based on that slice position
3. Combine predictions by averaging across all weak learners

## Results

Example results on IXI dataset (563 subjects, 80/20 train/test split):

| Method | MAE (years) | RMSE (years) | Correlation |
|--------|-------------|--------------|-------------|
| Early Fusion | ~7-8 | ~12-14 | ~0.75-0.80 |
| Late Fusion | ~7-9 | ~12-15 | ~0.70-0.78 |

*Results may vary based on random split and hyperparameters.*

## Output Files

- `brain_age_predictions.csv` - Individual subject predictions
- `brain_age_summary.csv` - Summary metrics
- `age_distributions.png` - Train/test age distribution plot
- `brain_age_results.png` - Prediction scatter plots and error distributions

## Citation

If you use this code, please cite:

```bibtex
@software{brain_age_estimation,
  title = {Brain Age Estimation from MRI},
  author = {Tanchaud},
  year = {2024},
  url = {https://github.com/tanchaud/brain-age-estimation}
}
```

## License

This project is available for academic and research purposes.

## Acknowledgments

- [IXI Dataset](https://brain-development.org/ixi-dataset/) for providing the brain MRI data
- VGG16 model from [Keras Applications](https://keras.io/api/applications/vgg/)
