import os
import numpy as np
import scipy.io
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
DATA_BASE_DIR = '/Users/tanchaud/Academia/Research_Project/BrainAge/Data' # Adjust this
NIFTI_TOOLBOX_PATH = '~/libs/nii' # Not directly used in Python if using nibabel
MATCONVNET_PATH = '~/libs/matconvnet-1.0-beta23' # Not directly used; Keras models are used instead

# Example: IXI dataset paths
IXI_DATA_DIR = os.path.join(DATA_BASE_DIR, 'brainage_new', 'IXI')
IXI_AGES_FILE = os.path.join(IXI_DATA_DIR, 'tables', 'AGE_IXI547.mat') # Assuming .mat, adjust if different
IXI_RIDS_FILE = os.path.join(IXI_DATA_DIR, 'tables', 'RID_IXI547.mat') # Assuming .mat

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


def main_workflow():
    print("--- Brain Age Project Workflow ---")

    # --- 1. Data Loading (Example with IXI dataset) ---
    print("\n--- 1. Data Loading ---")
    # The MATLAB script loads a small number `n=5` for initial testing.
    # Adjust `num_subjects_to_load` as needed.
    num_subjects_to_load = 5 # Small number for quick test
    
    # Get list of .nii files
    try:
        ixi_nii_files_all = [f for f in os.listdir(IXI_DATA_DIR) if f.endswith('.nii') or f.endswith('.nii.gz')]
        if not ixi_nii_files_all:
            print(f"No .nii files found in {IXI_DATA_DIR}. Skipping data loading.")
            return
        
        # Ensure we don't try to load more files than available
        actual_num_to_load = min(num_subjects_to_load, len(ixi_nii_files_all))
        if actual_num_to_load == 0 and num_subjects_to_load > 0 :
             print(f"Warning: No NIFTI files found to load in {IXI_DATA_DIR}")
             return
        elif actual_num_to_load < num_subjects_to_load:
             print(f"Warning: Requested {num_subjects_to_load} files, but only {actual_num_to_load} found/available.")

        ixi_nii_file_list_subset = ixi_nii_files_all[:actual_num_to_load]

        # Load NIFTI images
        # The load_nifti_slices function expects a list of filenames and the directory.
        X_ixi_mri_volumes = load_nifti_slices(IXI_DATA_DIR, ixi_nii_file_list_subset, n=actual_num_to_load)
        if not X_ixi_mri_volumes:
            print("Failed to load MRI volumes.")
            return
        print(f"Loaded {len(X_ixi_mri_volumes)} IXI MRI volumes.")

        # Load corresponding ages and RIDs
        # These files might need specific keys to extract data.
        ages_data = scipy.io.loadmat(IXI_AGES_FILE)
        # The MATLAB code `Y_ixi = Y_ixi.age_IXI547(1:n);` implies `age_IXI547` is the key.
        Y_ixi_ages_all = ages_data['age_IXI547'].flatten() # Adjust key if necessary
        Y_ixi_ages = Y_ixi_ages_all[:actual_num_to_load]

        rids_data = scipy.io.loadmat(IXI_RIDS_FILE)
        # The MATLAB code `I_ixi = I_ixi.rid(1:n);` implies `rid` is the key.
        I_ixi_rids_all = rids_data['RID_IXI'].flatten() # Adjust key 'rid' or 'RID_IXI' as in your file
        I_ixi_rids = I_ixi_rids_all[:actual_num_to_load]
        
        print(f"Loaded {len(Y_ixi_ages)} ages and {len(I_ixi_rids)} RIDs.")

    except FileNotFoundError as e:
        print(f"Error: Required data file not found: {e}. Please check paths.")
        print(f"Attempted to use IXI_DATA_DIR: {IXI_DATA_DIR}")
        print(f"Attempted age file: {IXI_AGES_FILE}, RID file: {IXI_RIDS_FILE}")
        print("Please create dummy files or point to correct data for testing if actual data is not present.")
        # Create dummy data for workflow to proceed if files are missing
        print("Creating dummy data to proceed with the workflow structure.")
        actual_num_to_load = num_subjects_to_load
        X_ixi_mri_volumes = [np.random.rand(64, 64, 40) for _ in range(actual_num_to_load)] # Dummy image volumes
        Y_ixi_ages = np.random.randint(20, 80, actual_num_to_load)
        I_ixi_rids = np.arange(100, 100 + actual_num_to_load)
        print(f"Using dummy data: {actual_num_to_load} subjects.")

    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return
        
    if not X_ixi_mri_volumes or Y_ixi_ages.size == 0 or I_ixi_rids.size == 0:
        print("Data loading failed or resulted in empty data. Aborting.")
        return


    # --- 2. Data Splitting ---
    print("\n--- 2. Data Splitting ---")
    # `D = Data_split(X_ixi,Y_ixi,I_ixi);`
    # `data_split` expects MRI data as the first argument.
    split_data_dict = data_split(X_ixi_mri_volumes, Y_ixi_ages, I_ixi_rids)
    print("Data split into Training (TS) and Validation/Test (VS) sets.")
    print(f"  Training samples: {len(split_data_dict['TS_labels'])}")
    print(f"  Validation/Test samples: {len(split_data_dict['VS_labels'])}")

    # Plot data distributions (optional, like `data_dist` in MATLAB)
    fig_dist, ax_dist = plt.subplots()
    plot_data_distribution(Y_ixi_ages, 'cyan', 'Original IXI Ages', ax_dist)
    plot_data_distribution(split_data_dict['TS_labels'], 'blue', 'TS Ages', ax_dist)
    plot_data_distribution(split_data_dict['VS_labels'], 'green', 'VS Ages', ax_dist)
    ax_dist.legend()
    plt.title("Age Distributions")
    # plt.show() # Show interactively or save
    plt.savefig("age_distributions.png")
    print("Saved age distribution plot to age_distributions.png")
    plt.close(fig_dist)


    # --- 3. Feature Extraction ---
    print("\n--- 3. Feature Extraction ---")
    if cnn_base_model is None or FEATURE_EXTRACTOR_TARGET_LAYER is None:
        print("CNN model not available. Skipping feature extraction.")
        # Create dummy features to allow rest of workflow to proceed structurally
        # This is for illustration. In a real run, feature extraction is crucial.
        print("Creating dummy features for TS and VS to continue workflow demonstration.")
        dummy_feature_dim = 100 # Arbitrary feature dimension for dummy data
        # TS features
        ts_image_volumes_from_split = split_data_dict['TS'] # list of image volumes
        X_ts_features = [np.random.rand(img.shape[2]//3 if img.shape[2]//3 > 0 else 1, dummy_feature_dim) for img in ts_image_volumes_from_split] # list of (n_triplets, feat_dim)
        # VS features
        vs_image_volumes_from_split = split_data_dict['VS']
        X_vs_features = [np.random.rand(img.shape[2]//3 if img.shape[2]//3 > 0 else 1, dummy_feature_dim) for img in vs_image_volumes_from_split]

    else:
        # Extract features for the Training Set images from the split
        # The `split_data_dict['TS']` contains the image volumes for training.
        print("Extracting features for Training Set (TS)...")
        ts_image_volumes_from_split = split_data_dict['TS'] # This is a list of MRI volumes
        X_ts_features = extract_cnn_features_non_overlapping_triplets(
            ts_image_volumes_from_split,
            cnn_base_model,
            FEATURE_EXTRACTOR_TARGET_LAYER
        )
        print(f"Extracted features for {len(X_ts_features)} TS subjects.")
        if X_ts_features and X_ts_features[0].size > 0 :
            print(f"  Feature shape for first TS subject: {X_ts_features[0].shape} (num_triplets, feature_dim)")
        else:
            print(f"  No features extracted or first subject had no features.")


        # Extract features for the Validation/Test Set images from the split
        print("Extracting features for Validation/Test Set (VS)...")
        vs_image_volumes_from_split = split_data_dict['VS']
        X_vs_features = extract_cnn_features_non_overlapping_triplets(
            vs_image_volumes_from_split,
            cnn_base_model,
            FEATURE_EXTRACTOR_TARGET_LAYER
        )
        print(f"Extracted features for {len(X_vs_features)} VS subjects.")
        if X_vs_features and X_vs_features[0].size > 0:
            print(f"  Feature shape for first VS subject: {X_vs_features[0].shape}")
        else:
             print(f"  No features extracted or first subject had no features for VS.")


    # Labels for training and validation sets
    Y_ts_labels = split_data_dict['TS_labels']
    Y_vs_labels = split_data_dict['VS_labels']


    # --- 4. Fusion Schemes ---

    # --- 4.A. Early Fusion ---
    # Early fusion typically requires a single feature vector per subject.
    # The `X_ts_features` (and `X_vs_features`) is a list of arrays,
    # where each array is (num_triplets, feature_dimension).
    # We need to aggregate these triplet features into one vector per subject.
    # Common aggregation: mean, max, or concatenate (if fixed num_triplets).
    # Let's use mean for this example.
    print("\n--- 4.A. Early Fusion (using mean of triplet features) ---")

    if not X_ts_features or not any(f.size > 0 for f in X_ts_features) :
        print("TS features are empty or invalid. Skipping Early Fusion for TS.")
    else:
        X_ts_aggregated_features = np.array([np.mean(f, axis=0) if f.size > 0 else np.zeros(X_ts_features[0].shape[1] if X_ts_features[0].size > 0 else 1) for f in X_ts_features])
        print(f"Aggregated TS features shape for Early Fusion: {X_ts_aggregated_features.shape}")
        
        # Train and test Early Fusion model on Training Set data itself (as an example)
        # Or, if you have separate Test set features, use those.
        # The MATLAB 'Early_Fusion.m' script loads various pre-computed features and labels.
        # Here, we use the ones derived from our current split.
        print("Running Early Fusion on Training Set data (train and predict on same for demo):")
        # Note: The MATLAB script tests on the same data it might have trained on, or loads separate test data.
        # For a proper evaluation, train on TS, test on VS.
        # early_fusion_train_test(X_features_arg=X_ts_aggregated_features, Y_labels_arg=Y_ts_labels)
        
        # For a more standard approach: Train on TS, Test on VS
        if not X_vs_features or not any(f.size > 0 for f in X_vs_features):
            print("VS features are empty or invalid. Skipping Early Fusion evaluation on VS.")
        else:
            X_vs_aggregated_features = np.array([np.mean(f, axis=0) if f.size > 0 else np.zeros(X_vs_features[0].shape[1] if X_vs_features[0].size > 0 else 1) for f in X_vs_features])
            print(f"Aggregated VS features shape for Early Fusion: {X_vs_aggregated_features.shape}")

            print("\nTraining Early Fusion model on TS, evaluating on VS:")
            # Create and train the SVR model
            # Parameters C=98, epsilon=0.064 from an example in Early_Fusion.m
            early_fusion_model = SVR(kernel='linear', C=98, epsilon=0.064)
            early_fusion_model.fit(X_ts_aggregated_features, Y_ts_labels)
            
            # Predict on VS
            y_pred_ef_vs = early_fusion_model.predict(X_vs_aggregated_features)
            
            # Evaluate
            mae_ef_vs = mae(Y_vs_labels, y_pred_ef_vs)
            rmse_ef_vs = rmse(Y_vs_labels, y_pred_ef_vs)
            corr_ef_vs = pearson_correlation(Y_vs_labels, y_pred_ef_vs)
            
            print(f"  MAE on VS (Early Fusion): {mae_ef_vs:.4f}")
            print(f"  RMSE on VS (Early Fusion): {rmse_ef_vs:.4f}")
            print(f"  Correlation on VS (Early Fusion): {corr_ef_vs:.4f}")


    # --- 4.B. Late Fusion ---
    print("\n--- 4.B. Late Fusion ---")
    # Late fusion uses features from each modality (e.g., triplet) separately.
    # `X_ts_features` is already in the format: list of (num_triplets, feature_dim) per patient.
    # `train_and_predict_weak_learners` expects this format.

    # For Late Fusion, typically train weak learners on TS, then apply to VS, then combine on VS.
    # Or, train and combine on TS itself for an internal check.
    # The MATLAB script seems to evaluate on the same set it used for training weak learners.
    # Let's use VS data for demonstration of evaluating late fusion.
    
    if not X_vs_features or not Y_vs_labels.size > 0 or not any(f.size > 0 for f in X_vs_features):
        print("VS features or labels are empty/invalid. Skipping Late Fusion.")
    else:
        print("Training/evaluating Late Fusion on Validation Set (VS) data:")
        # The function train_and_predict_weak_learners will train new models or load them.
        # It will predict on the same data (X_vs_features) for this example flow.
        # In a strict train/test split, weak learners are trained on TS features/labels,
        # then these trained models predict on VS features.
        # For simplicity here, analogous to parts of the MATLAB script,
        # we demonstrate the process on one dataset (VS).
        
        # Step 1: Train weak learners on TS and get their predictions on VS
        # This requires adapting `train_and_predict_weak_learners` or a new function.
        # Let's assume `train_and_predict_weak_learners` is modified or used as follows:
        #   - Train N SVRs (one for each triplet type) using all TS patient data for that triplet type.
        #   - Then, for each VS patient, get N predictions using these N SVRs.

        # Simplified: Perform weak learning and prediction generation on VS set directly for demo
        # This is not a typical train/test evaluation but matches some script structures.
        print("Generating weak scores on VS data (models trained and predict on VS subsets)...")
        # `X_vs_features` is list of [n_triplets, feat_dim_per_triplet]
        # `Y_vs_labels` are the true ages for these VS patients
        weak_scores_vs = train_and_predict_weak_learners(X_vs_features, Y_vs_labels)
                                                        # model_save_dir="weak_models_vs_svr")

        if weak_scores_vs.size == 0:
            print("Failed to generate weak scores for VS. Skipping Late Fusion combination.")
        else:
            print(f"Shape of weak_scores_vs: {weak_scores_vs.shape} (expected: num_triplets, num_vs_patients)")

            # Step 2: Combine weak scores
            # `weak_scores_vs` is (num_triplets, num_vs_patients)
            # `Y_vs_labels` is (num_vs_patients,)
            num_total_triplets_modalities = weak_scores_vs.shape[0]
            num_vs_patients = weak_scores_vs.shape[1]

            combined_predictions_vs = combine_weak_predictions(
                weak_scores_vs,
                Y_vs_labels,
                num_total_triplets_modalities,
                num_vs_patients
            )

            # Step 3: Evaluate combined predictions
            print("\nLate Fusion Performance on VS:")
            for strategy, y_pred_lf in combined_predictions_vs.items():
                if y_pred_lf is not None and not np.all(np.isnan(y_pred_lf)) : # Check if predictions are valid
                    mae_lf = mae(Y_vs_labels, y_pred_lf)
                    rmse_lf = rmse(Y_vs_labels, y_pred_lf)
                    corr_lf = pearson_correlation(Y_vs_labels, y_pred_lf)
                    print(f"  Strategy: {strategy}")
                    print(f"    MAE: {mae_lf:.4f}, RMSE: {rmse_lf:.4f}, Correlation: {corr_lf:.4f}")
                else:
                    print(f"  Strategy: {strategy} - Predictions are NaN or None.")

    print("\n--- Workflow Complete ---")


if __name__ == '__main__':
    # Setup environment (e.g., for TensorFlow GPU, logging) if needed
    # import tensorflow as tf
    # print(f"TensorFlow version: {tf.__version__}")
    # print(f"Num GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))}")

    # Create dummy directories and files if they don't exist for the example to run
    # This is for the IXI data loading part.
    # In a real scenario, these paths should point to your actual data.
    if not os.path.exists(IXI_DATA_DIR):
        os.makedirs(os.path.join(IXI_DATA_DIR, 'tables'), exist_ok=True)
        print(f"Created dummy directory structure: {IXI_DATA_DIR}")
        # Create dummy .nii files
        for i in range(10): # Create 10 dummy NIFTI files
            try:
                import nibabel as nib
                dummy_nii_data = np.random.rand(10,10,120).astype(np.float32)
                nif = nib.Nifti1Image(dummy_nii_data, np.eye(4))
                nib.save(nif, os.path.join(IXI_DATA_DIR, f'dummy_ixi_{i}.nii'))
            except ImportError:
                print("nibabel not installed, cannot create dummy .nii files. Skipping.")
                break
            except Exception as e_nii:
                print(f"Error creating dummy nii: {e_nii}")


        # Create dummy .mat files for ages and RIDs
        # Note: `AGE_IXI547` and `RID_IXI547` are from the original file names.
        # The script expects keys like 'age_IXI547' and 'RID_IXI'.
        num_dummy_subjects_for_mat = 547 # To match file name
        dummy_ages_mat = {'age_IXI547': np.random.randint(20, 85, num_dummy_subjects_for_mat)}
        dummy_rids_mat = {'RID_IXI': np.arange(1, num_dummy_subjects_for_mat + 1)}
        try:
            scipy.io.savemat(IXI_AGES_FILE, dummy_ages_mat)
            scipy.io.savemat(IXI_RIDS_FILE, dummy_rids_mat)
            print(f"Created dummy '{IXI_AGES_FILE}' and '{IXI_RIDS_FILE}'.")
        except Exception as e_mat:
            print(f"Error creating dummy .mat files: {e_mat}")


    main_workflow()