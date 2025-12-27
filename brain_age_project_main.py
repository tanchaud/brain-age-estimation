import os

import numpy as np
import pandas as pd
from sklearn import svm

import matplotlib.pyplot as plt # For data distribution plotting

# Import previously defined functions
# (Assuming they are in .py files in the same directory or accessible via PYTHONPATH)
from load_nifti import load_nifti_slices
from data_split import data_split
from feature_extraction import extract_cnn_features_non_overlapping_triplets
from early_fusion import early_fusion_train_test
from late_fusion import train_and_predict_weak_learners
from concat_predictions import combine_weak_predictions, placeholder_bt_analysis
from mae import mae
from rmse import rmse
from pearson_corr import pearson_correlation

# TensorFlow for CNN model (used in feature_extraction.py)
from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.applications.vgg19 import VGG19 # if using VGG19

# --- Configuration ---
# Set up your paths and model choices here
DATA_BASE_DIR = '/Users/tanchaud/Data_IXI/June' # Adjust this

# Example: IXI dataset paths
IXI_IMAGE_DIR = os.path.join(DATA_BASE_DIR,'IXI-T1')
IXI_DEMOGRAPHICS_FILE = os.path.join(DATA_BASE_DIR,'IXI.xls')

# Feature extraction CNN model (using Keras VGG16 as an example)
# The MATLAB code mentions: imagenet-vgg-f.mat, imagenet-vgg-verydeep-19.mat, imagenet-vgg-verydeep-16.mat
# We'll use VGG16.
try:
    cnn_base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # Choose the layer for feature extraction. This needs to map to what `res(16).x` or similar
    # in the MatConvNet script `Feat_Ext_130917.m` was aiming for.
    # Example: 'block5_pool'. If the target was 9216 features, this layer might need adjustment
    # or a different model/input size.
    FEATURE_EXTRACTOR_TARGET_LAYER = 'block5_pool'
except Exception as e:
    print(f"Could not load Keras VGG16 model. Ensure TensorFlow is installed and you have internet for weights download: {e}")
    cnn_base_model = None
    FEATURE_EXTRACTOR_TARGET_LAYER = None


# Placeholder for data_dist plotting function
def plot_data_distribution(data, color, label, ax):
    """Placeholder to plot data distribution (e.g., histogram)."""
    if ax is None:
        fig, ax = plt.subplots()
    ax.hist(data, bins=30, alpha=0.7, color=color, label=label, density=True)
    ax.set_xlabel("Age")
    ax.set_ylabel("Density")
    return ax


def extract_features_for_single_subject(file_path, cnn_model, target_layer, feature_extractor, num_slices=120):
    """
    Load a single MRI volume, extract features, and return both aggregated and per-triplet features.
    This is memory-efficient as it processes one subject at a time.

    Returns:
        tuple: (aggregated_features, triplet_features_array)
               - aggregated_features: mean of triplet features (for early fusion)
               - triplet_features_array: (num_triplets, feature_dim) array (for late fusion)
    """
    import nibabel as nib
    from tensorflow.keras.applications.vgg16 import preprocess_input
    import cv2

    # Load single NIFTI file
    nii_img = nib.load(file_path)
    mri_vol = nii_img.get_fdata()[:, :, :num_slices]

    input_size = (cnn_model.input_shape[1], cnn_model.input_shape[2])
    num_slices_vol = mri_vol.shape[2]
    total_triplets = num_slices_vol // 3

    triplet_features = []
    for k in range(total_triplets):
        idx = k * 3
        s1, s2, s3 = mri_vol[:, :, idx], mri_vol[:, :, idx+1], mri_vol[:, :, idx+2]
        triplet_img = np.stack((s1, s2, s3), axis=-1)
        resized = cv2.resize(triplet_img.astype(np.float32),
                            (input_size[1], input_size[0]),
                            interpolation=cv2.INTER_LINEAR)
        input_batch = np.expand_dims(resized, axis=0)
        preprocessed = preprocess_input(input_batch)
        features = feature_extractor.predict(preprocessed, verbose=0)
        triplet_features.append(features.flatten())

    if triplet_features:
        triplet_array = np.array(triplet_features)
        aggregated = np.mean(triplet_array, axis=0)
        return aggregated, triplet_array
    return None, None


def main_workflow():
    print("--- Brain Age Project Workflow ---")

    # --- 1. Data Loading (Metadata only - memory efficient) ---
    print("\n--- 1. Data Loading (Metadata) ---")

    # Number of subjects to process (set to None or large number for all)
    num_subjects_to_load = 590  # All subjects

    try:
        # Load and clean demographics dataframe
        df_demographics = pd.read_excel(IXI_DEMOGRAPHICS_FILE, sheet_name=0)
        df_demographics_cleaned = df_demographics.dropna(subset=['AGE']).copy()
        df_demographics_cleaned['IXI_ID'] = df_demographics_cleaned['IXI_ID'].astype(int)

        print(f"Demographics loaded: {len(df_demographics_cleaned)} subjects with valid ages")

        # Get list of .nii files and extract subject IDs from filenames
        ixi_nii_files_all = [f for f in os.listdir(IXI_IMAGE_DIR) if f.endswith('.nii') or f.endswith('.nii.gz')]
        if not ixi_nii_files_all:
            print(f"No .nii files found in {IXI_IMAGE_DIR}. Aborting.")
            return

        # Build a mapping of subject ID to filename
        file_mapping = {}
        for f in ixi_nii_files_all:
            ixi_id = int(f.split('-')[0].replace('IXI', ''))
            file_mapping[ixi_id] = f

        # Get subjects that have both MRI and demographics
        available_ids = set(file_mapping.keys()) & set(df_demographics_cleaned['IXI_ID'].values)
        available_ids = sorted(list(available_ids))[:num_subjects_to_load]

        print(f"Found {len(available_ids)} subjects with both MRI and demographics data")

        # Get ages for available subjects (in order)
        df_available = df_demographics_cleaned[df_demographics_cleaned['IXI_ID'].isin(available_ids)]
        df_available = df_available.set_index('IXI_ID').loc[available_ids].reset_index()

        Y_all_ages = df_available['AGE'].values
        I_all_ids = df_available['IXI_ID'].values

    except FileNotFoundError as e:
        print(f"Error: Required data file not found: {e}.")
        return
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return

    # --- 2. Data Splitting (by indices, not loaded data) ---
    print("\n--- 2. Data Splitting ---")
    n_subjects = len(available_ids)
    indices = np.arange(n_subjects)
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(indices)

    # 80-20 split
    split_idx = int(0.8 * n_subjects)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    Y_ts_labels = Y_all_ages[train_indices]
    Y_vs_labels = Y_all_ages[test_indices]
    train_ids = I_all_ids[train_indices]
    test_ids = I_all_ids[test_indices]

    print(f"  Training samples: {len(train_indices)}")
    print(f"  Validation/Test samples: {len(test_indices)}")

    # Plot data distributions
    fig_dist, ax_dist = plt.subplots()
    plot_data_distribution(Y_all_ages, 'cyan', 'All Ages', ax_dist)
    plot_data_distribution(Y_ts_labels, 'blue', 'TS Ages', ax_dist)
    plot_data_distribution(Y_vs_labels, 'green', 'VS Ages', ax_dist)
    ax_dist.legend()
    plt.title("Age Distributions")
    plt.savefig("age_distributions.png")
    print("Saved age distribution plot to age_distributions.png")
    plt.close(fig_dist)

    # --- 3. Feature Extraction (one subject at a time - memory efficient) ---
    print("\n--- 3. Feature Extraction (memory-efficient, one subject at a time) ---")

    if cnn_base_model is None or FEATURE_EXTRACTOR_TARGET_LAYER is None:
        print("CNN model not available. Aborting.")
        return

    # Create feature extractor once (more efficient than recreating in loop)
    from tensorflow.keras.models import Model
    feature_extractor = Model(inputs=cnn_base_model.input,
                              outputs=cnn_base_model.get_layer(FEATURE_EXTRACTOR_TARGET_LAYER).output)

    # Extract training features
    print(f"Extracting features for {len(train_ids)} training subjects...")
    X_ts_aggregated = []
    X_ts_triplets = []  # For late fusion
    for i, subj_id in enumerate(train_ids):
        file_path = os.path.join(IXI_IMAGE_DIR, file_mapping[subj_id])
        try:
            agg_feat, triplet_feat = extract_features_for_single_subject(
                file_path, cnn_base_model, FEATURE_EXTRACTOR_TARGET_LAYER, feature_extractor
            )
            if agg_feat is not None:
                X_ts_aggregated.append(agg_feat)
                X_ts_triplets.append(triplet_feat)
            else:
                X_ts_aggregated.append(np.zeros(25088))
                X_ts_triplets.append(np.zeros((40, 25088)))
        except Exception as e:
            print(f"  Error processing subject {subj_id}: {e}")
            X_ts_aggregated.append(np.zeros(25088))
            X_ts_triplets.append(np.zeros((40, 25088)))

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Processed {i + 1}/{len(train_ids)} training subjects", flush=True)

    X_ts_aggregated_features = np.array(X_ts_aggregated)
    print(f"Training features shape: {X_ts_aggregated_features.shape}")

    # Extract test features
    print(f"\nExtracting features for {len(test_ids)} test subjects...")
    X_vs_aggregated = []
    X_vs_triplets = []  # For late fusion
    for i, subj_id in enumerate(test_ids):
        file_path = os.path.join(IXI_IMAGE_DIR, file_mapping[subj_id])
        try:
            agg_feat, triplet_feat = extract_features_for_single_subject(
                file_path, cnn_base_model, FEATURE_EXTRACTOR_TARGET_LAYER, feature_extractor
            )
            if agg_feat is not None:
                X_vs_aggregated.append(agg_feat)
                X_vs_triplets.append(triplet_feat)
            else:
                X_vs_aggregated.append(np.zeros(25088))
                X_vs_triplets.append(np.zeros((40, 25088)))
        except Exception as e:
            print(f"  Error processing subject {subj_id}: {e}")
            X_vs_aggregated.append(np.zeros(25088))
            X_vs_triplets.append(np.zeros((40, 25088)))

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Processed {i + 1}/{len(test_ids)} test subjects", flush=True)

    X_vs_aggregated_features = np.array(X_vs_aggregated)
    print(f"Test features shape: {X_vs_aggregated_features.shape}")


    # --- 4. Early Fusion (SVR on aggregated features) ---
    print("\n--- 4. Early Fusion ---")
    print(f"Training features shape: {X_ts_aggregated_features.shape}")
    print(f"Test features shape: {X_vs_aggregated_features.shape}")

    print("\nTraining Early Fusion SVR model...")
    # Parameters C=98, epsilon=0.064 from Early_Fusion.m
    early_fusion_model = svm.SVR(kernel='linear', C=98, epsilon=0.064)
    early_fusion_model.fit(X_ts_aggregated_features, Y_ts_labels)

    # Predict on validation set
    y_pred_vs = early_fusion_model.predict(X_vs_aggregated_features)

    # Evaluate
    mae_val = mae(Y_vs_labels, y_pred_vs)
    rmse_val = rmse(Y_vs_labels, y_pred_vs)
    corr_val = pearson_correlation(Y_vs_labels, y_pred_vs)

    print("\n" + "="*50)
    print("RESULTS - Early Fusion (SVR)")
    print("="*50)
    print(f"  MAE:  {mae_val:.2f} years")
    print(f"  RMSE: {rmse_val:.2f} years")
    print(f"  Correlation: {corr_val:.4f}")
    print("="*50)

    # Save Early Fusion predictions
    results_df = pd.DataFrame({
        'Subject_ID': test_ids,
        'True_Age': Y_vs_labels,
        'Predicted_Age_EarlyFusion': y_pred_vs,
        'Error_EarlyFusion': y_pred_vs - Y_vs_labels
    })

    # --- 5. Late Fusion ---
    print("\n--- 5. Late Fusion ---")
    print("Training weak learners (one SVR per triplet modality)...")

    # Late fusion: train one SVR per triplet position using training data
    # Then predict on test data and combine predictions
    num_triplets = X_ts_triplets[0].shape[0]  # e.g., 40 triplets
    num_train = len(X_ts_triplets)
    num_test = len(X_vs_triplets)

    print(f"  Number of triplet modalities: {num_triplets}")
    print(f"  Training subjects: {num_train}, Test subjects: {num_test}")

    # Reorganize data: for each triplet position, collect features from all subjects
    # weak_predictions[triplet_idx, test_subject_idx] = predicted age
    weak_predictions = np.zeros((num_triplets, num_test))

    for t in range(num_triplets):
        # Get features for triplet t from all training subjects
        X_train_t = np.array([X_ts_triplets[j][t, :] for j in range(num_train)])
        # Get features for triplet t from all test subjects
        X_test_t = np.array([X_vs_triplets[j][t, :] for j in range(num_test)])

        # Train SVR for this triplet
        weak_svr = svm.SVR(kernel='linear', C=1.0, epsilon=0.1)
        weak_svr.fit(X_train_t, Y_ts_labels)

        # Predict on test data
        weak_predictions[t, :] = weak_svr.predict(X_test_t)

        if (t + 1) % 10 == 0 or t == 0:
            print(f"  Trained weak learner {t + 1}/{num_triplets}", flush=True)

    print(f"Weak predictions shape: {weak_predictions.shape}")

    # Combine predictions using different strategies
    # Strategy 1: Simple mean
    y_pred_late_mean = np.mean(weak_predictions, axis=0)

    # Strategy 2: Weighted mean (weight by inverse MAE on training set)
    # For simplicity, use equal weights here (same as mean)
    y_pred_late_wmean = y_pred_late_mean  # Could implement weighted version

    # Evaluate Late Fusion
    mae_late = mae(Y_vs_labels, y_pred_late_mean)
    rmse_late = rmse(Y_vs_labels, y_pred_late_mean)
    corr_late = pearson_correlation(Y_vs_labels, y_pred_late_mean)

    print("\n" + "="*50)
    print("RESULTS - Late Fusion (Mean of Weak Learners)")
    print("="*50)
    print(f"  MAE:  {mae_late:.2f} years")
    print(f"  RMSE: {rmse_late:.2f} years")
    print(f"  Correlation: {corr_late:.4f}")
    print("="*50)

    # Add Late Fusion results to dataframe
    results_df['Predicted_Age_LateFusion'] = y_pred_late_mean
    results_df['Error_LateFusion'] = y_pred_late_mean - Y_vs_labels

    # Summary comparison
    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("="*50)
    print(f"{'Method':<20} {'MAE':>10} {'RMSE':>10} {'Corr':>10}")
    print("-"*50)
    print(f"{'Early Fusion':<20} {mae_val:>10.2f} {rmse_val:>10.2f} {corr_val:>10.4f}")
    print(f"{'Late Fusion':<20} {mae_late:>10.2f} {rmse_late:>10.2f} {corr_late:>10.4f}")
    print("="*50)

    # Save all predictions
    results_df.to_csv('brain_age_predictions.csv', index=False)
    print("\nAll predictions saved to brain_age_predictions.csv")

    print("\n--- Workflow Complete ---")


if __name__ == '__main__':
    # Setup environment (e.g., for TensorFlow GPU, logging) if needed
    # import tensorflow as tf
    # print(f"TensorFlow version: {tf.__version__}")
    # print(f"Num GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))}")

    # Create dummy directories and files if they don't exist for the example to run
    # This is for the IXI data loading part.
    # In a real scenario, these paths should point to your actual data.
    if not os.path.exists(IXI_IMAGE_DIR):
        os.makedirs(IXI_IMAGE_DIR, exist_ok=True)
        print(f"Created dummy directory structure: {IXI_IMAGE_DIR}")
        # Create dummy .nii files with IXI naming convention
        for i in range(10):
            try:
                import nibabel as nib
                dummy_nii_data = np.random.rand(10, 10, 120).astype(np.float32)
                nif = nib.Nifti1Image(dummy_nii_data, np.eye(4))
                # Use IXI naming format: IXI{ID}-Site-Session-T1.nii.gz
                filename = f'IXI{i:03d}-Guys-0001-T1.nii.gz'
                nib.save(nif, os.path.join(IXI_IMAGE_DIR, filename))
            except ImportError:
                print("nibabel not installed, cannot create dummy .nii files. Skipping.")
                break
            except Exception as e_nii:
                print(f"Error creating dummy nii: {e_nii}")


        # Create dummy demographics Excel file
        try:
            dummy_ixi_ids = list(range(10))
            dummy_ages = np.random.randint(20, 85, 10)
            df_dummy = pd.DataFrame({'IXI_ID': dummy_ixi_ids, 'AGE': dummy_ages})
            df_dummy.to_excel(IXI_DEMOGRAPHICS_FILE, index=False)
            print(f"Created dummy demographics file: '{IXI_DEMOGRAPHICS_FILE}'.")
        except Exception as e_xls:
            print(f"Error creating dummy Excel file: {e_xls}")


    main_workflow()