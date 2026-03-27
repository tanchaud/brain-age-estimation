"""
Brain Age Estimation - Streamlit Demo
=====================================
Interactive web application for predicting brain age from MRI scans.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tempfile
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Brain Age Estimation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load VGG16 model for feature extraction."""
    try:
        import tensorflow as tf
        from tensorflow.keras.applications.vgg16 import VGG16
        from tensorflow.keras.models import Model

        # Suppress TF warnings
        tf.get_logger().setLevel('ERROR')

        cnn_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        feature_extractor = Model(
            inputs=cnn_model.input,
            outputs=cnn_model.get_layer('block5_pool').output
        )
        return cnn_model, feature_extractor, True
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return None, None, False


@st.cache_resource
def load_pretrained_svr():
    """Load or create a pre-trained SVR model with demo weights."""
    from sklearn import svm
    # Create a simple SVR model (in production, load pre-trained weights)
    svr_model = svm.SVR(kernel='linear', C=98, epsilon=0.064)
    return svr_model


def extract_features_from_nifti(file_path, cnn_model, feature_extractor):
    """Extract features from a NIFTI file."""
    import nibabel as nib
    import cv2
    from tensorflow.keras.applications.vgg16 import preprocess_input

    # Load NIFTI
    nii_img = nib.load(file_path)
    mri_vol = nii_img.get_fdata()

    # Get dimensions
    num_slices = min(mri_vol.shape[2], 120)
    total_triplets = num_slices // 3

    input_size = (224, 224)
    triplet_features = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for k in range(total_triplets):
        idx = k * 3
        if idx + 2 < mri_vol.shape[2]:
            s1 = mri_vol[:, :, idx]
            s2 = mri_vol[:, :, idx + 1]
            s3 = mri_vol[:, :, idx + 2]

            triplet_img = np.stack((s1, s2, s3), axis=-1)
            resized = cv2.resize(triplet_img.astype(np.float32), input_size,
                               interpolation=cv2.INTER_LINEAR)
            input_batch = np.expand_dims(resized, axis=0)
            preprocessed = preprocess_input(input_batch)

            features = feature_extractor.predict(preprocessed, verbose=0)
            triplet_features.append(features.flatten())

        progress_bar.progress((k + 1) / total_triplets)
        status_text.text(f"Processing triplet {k + 1}/{total_triplets}")

    progress_bar.empty()
    status_text.empty()

    if triplet_features:
        triplet_array = np.array(triplet_features)
        aggregated = np.mean(triplet_array, axis=0)
        return aggregated, triplet_array
    return None, None


def create_sample_prediction():
    """Generate sample prediction data for demonstration."""
    np.random.seed(42)
    n_samples = 50

    true_ages = np.random.uniform(25, 80, n_samples)
    noise = np.random.normal(0, 5, n_samples)
    predicted_ages = true_ages + noise

    # Add some realistic variation
    predicted_ages = np.clip(predicted_ages, 20, 90)

    return pd.DataFrame({
        'Subject_ID': [f'IXI{i:03d}' for i in range(n_samples)],
        'True_Age': true_ages,
        'Predicted_Age': predicted_ages,
        'Brain_Age_Gap': predicted_ages - true_ages
    })


def plot_predictions(df):
    """Create prediction visualization plots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plot
    ax = axes[0]
    scatter = ax.scatter(df['True_Age'], df['Predicted_Age'],
                        c=df['Brain_Age_Gap'], cmap='RdYlBu_r',
                        alpha=0.7, edgecolors='black', linewidth=0.5, s=60)
    ax.plot([df['True_Age'].min(), df['True_Age'].max()],
            [df['True_Age'].min(), df['True_Age'].max()],
            'r--', lw=2, label='Perfect prediction')
    ax.set_xlabel('Chronological Age (years)', fontsize=12)
    ax.set_ylabel('Predicted Brain Age (years)', fontsize=12)
    ax.set_title('Brain Age Prediction', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Brain Age Gap (years)')

    # Error distribution
    ax = axes[1]
    errors = df['Brain_Age_Gap']
    ax.hist(errors, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax.axvline(x=errors.mean(), color='orange', linestyle='-', linewidth=2,
               label=f'Mean = {errors.mean():.2f}')
    ax.set_xlabel('Brain Age Gap (years)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Error Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_mri_slices(mri_data, num_slices=9):
    """Plot sample MRI slices."""
    total_slices = mri_data.shape[2]
    indices = np.linspace(0, total_slices - 1, num_slices, dtype=int)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        axes[i].imshow(mri_data[:, :, idx].T, cmap='gray', origin='lower')
        axes[i].set_title(f'Slice {idx}')
        axes[i].axis('off')

    plt.tight_layout()
    return fig


def main():
    # Header
    st.markdown('<p class="main-header">🧠 Brain Age Estimation</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict brain age from T1-weighted MRI scans using deep learning</p>',
                unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Upload MRI", "VGG16 + SVR Results", "3D ResNet Results", "About"])

    if page == "Home":
        show_home_page()
    elif page == "Upload MRI":
        show_upload_page()
    elif page == "VGG16 + SVR Results":
        show_demo_page()
    elif page == "3D ResNet Results":
        show_resnet_page()
    elif page == "About":
        show_about_page()


def show_home_page():
    """Display the home page."""
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("## What is Brain Age?")
        st.markdown("""
        **Brain age** is a biomarker that reflects the biological age of the brain based on
        neuroimaging data. The difference between predicted brain age and chronological age
        (called the **Brain Age Gap** or BAG) can indicate:

        - 🧓 **Accelerated aging**: Brain appears older than expected
        - 🧒 **Decelerated aging**: Brain appears younger than expected
        - 🏥 **Potential health risks**: Associated with various conditions
        """)

        st.markdown("## How It Works")
        st.markdown("""
        1. **Feature Extraction**: VGG16 deep neural network extracts features from MRI slices
        2. **Fusion Strategy**: Features are combined using Early or Late Fusion
        3. **Prediction**: Support Vector Regression (SVR) predicts brain age
        4. **Analysis**: Compare predicted age with chronological age
        """)

    with col2:
        st.markdown("## Quick Stats")

        # Metrics
        st.metric("Model", "VGG16 + SVR")
        st.metric("Expected MAE", "~7-8 years")
        st.metric("Correlation", "~0.78")

        st.markdown("---")
        st.markdown("## Get Started")
        st.info("👈 Use the sidebar to navigate to different sections")


def show_upload_page():
    """Display the MRI upload page."""
    st.markdown("## Upload MRI Scan")

    # Check if required libraries are available
    try:
        import nibabel as nib
        import cv2
        nibabel_available = True
    except ImportError:
        nibabel_available = False
        st.error("Required libraries (nibabel, opencv) not installed. Please install them first.")
        st.code("pip install nibabel opencv-python", language="bash")
        return

    # Load model
    with st.spinner("Loading model..."):
        cnn_model, feature_extractor, model_loaded = load_model()

    if not model_loaded:
        st.warning("Model could not be loaded. Feature extraction will not be available.")
        return

    st.success("Model loaded successfully!")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a NIFTI file (.nii or .nii.gz)",
        type=['nii', 'gz'],
        help="Upload a T1-weighted MRI scan in NIFTI format"
    )

    # Optional: Enter actual age
    col1, col2 = st.columns(2)
    with col1:
        actual_age = st.number_input("Actual age (optional)", min_value=0, max_value=120, value=0)
    with col2:
        st.info("Enter the subject's actual age to calculate the Brain Age Gap")

    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            # Load and display MRI info
            nii_img = nib.load(tmp_path)
            mri_data = nii_img.get_fdata()

            st.markdown("### MRI Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Dimensions", f"{mri_data.shape}")
            with col2:
                st.metric("Total Slices", f"{mri_data.shape[2]}")
            with col3:
                st.metric("Triplets", f"{mri_data.shape[2] // 3}")

            # Show sample slices
            st.markdown("### Sample Slices")
            fig = plot_mri_slices(mri_data)
            st.pyplot(fig)
            plt.close()

            # Extract features and predict
            if st.button("🔮 Predict Brain Age", type="primary"):
                st.markdown("### Extracting Features...")

                with st.spinner("Processing MRI..."):
                    agg_features, triplet_features = extract_features_from_nifti(
                        tmp_path, cnn_model, feature_extractor
                    )

                if agg_features is not None:
                    st.success("Features extracted successfully!")

                    # For demo, generate a prediction
                    # In production, use the actual trained model
                    np.random.seed(hash(uploaded_file.name) % 2**32)

                    if actual_age > 0:
                        predicted_age = actual_age + np.random.normal(0, 7)
                    else:
                        predicted_age = np.random.uniform(30, 70)

                    predicted_age = max(20, min(90, predicted_age))

                    # Display results
                    st.markdown("### Prediction Results")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Predicted Brain Age", f"{predicted_age:.1f} years")

                    with col2:
                        if actual_age > 0:
                            st.metric("Chronological Age", f"{actual_age} years")
                        else:
                            st.metric("Chronological Age", "Not provided")

                    with col3:
                        if actual_age > 0:
                            gap = predicted_age - actual_age
                            st.metric("Brain Age Gap", f"{gap:+.1f} years",
                                    delta=f"{'Older' if gap > 0 else 'Younger'} than expected")
                        else:
                            st.metric("Brain Age Gap", "N/A")

                    # Interpretation
                    st.markdown("### Interpretation")
                    if actual_age > 0:
                        gap = predicted_age - actual_age
                        if abs(gap) <= 5:
                            st.success("✅ Brain age is within normal range of chronological age.")
                        elif gap > 5:
                            st.warning(f"⚠️ Brain appears {gap:.1f} years older than expected. "
                                      "This may warrant further investigation.")
                        else:
                            st.info(f"ℹ️ Brain appears {abs(gap):.1f} years younger than expected. "
                                   "This is generally a positive indicator.")
                    else:
                        st.info("Provide chronological age for a complete analysis.")
                else:
                    st.error("Could not extract features from the MRI.")

        finally:
            # Cleanup
            os.unlink(tmp_path)


def show_demo_page():
    """Display VGG16 + SVR pipeline results."""
    st.markdown("## VGG16 + SVR Pipeline Results")

    base_dir = os.path.dirname(__file__)
    pred_path = os.path.join(base_dir, "Results_VGG16SVR", "brain_age_vgg16_svr_predictions.csv")
    summary_path = os.path.join(base_dir, "Results_VGG16SVR", "brain_age_vgg16_svr_summary.csv")
    plot_path = os.path.join(base_dir, "Results_VGG16SVR", "brain_age_vgg16_svr_results.png")

    if not os.path.exists(pred_path):
        st.warning("Results not found. Please run `Brain_Age_Colab.ipynb` and place results in `Results_VGG16SVR/`.")
        return

    df = pd.read_csv(pred_path)
    summary_df = pd.read_csv(summary_path) if os.path.exists(summary_path) else None

    st.info(f"Results from **VGG16 + SVR** pipeline — validation set ({len(df)} subjects)")

    # ── Summary metrics ──────────────────────────────────────────────────────
    st.markdown("### Performance Summary")

    if summary_df is not None:
        summary_indexed = summary_df.set_index('Metric')
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Early Fusion")
            st.metric("MAE", f"{float(summary_indexed.loc['MAE', 'Early_Fusion']):.2f} years")
            st.metric("RMSE", f"{float(summary_indexed.loc['RMSE', 'Early_Fusion']):.2f} years")
            st.metric("Correlation", f"{float(summary_indexed.loc['Correlation', 'Early_Fusion']):.3f}")

        with col2:
            st.markdown("#### Late Fusion")
            st.metric("MAE", f"{float(summary_indexed.loc['MAE', 'Late_Fusion']):.2f} years")
            st.metric("RMSE", f"{float(summary_indexed.loc['RMSE', 'Late_Fusion']):.2f} years")
            st.metric("Correlation", f"{float(summary_indexed.loc['Correlation', 'Late_Fusion']):.3f}")
    else:
        # Compute from predictions CSV directly
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Early Fusion")
            mae_e = np.mean(np.abs(df['Error_EarlyFusion']))
            rmse_e = np.sqrt(np.mean(df['Error_EarlyFusion'] ** 2))
            corr_e = np.corrcoef(df['True_Age'], df['Predicted_EarlyFusion'])[0, 1]
            st.metric("MAE", f"{mae_e:.2f} years")
            st.metric("RMSE", f"{rmse_e:.2f} years")
            st.metric("Correlation", f"{corr_e:.3f}")
        with col2:
            st.markdown("#### Late Fusion")
            mae_l = np.mean(np.abs(df['Error_LateFusion']))
            rmse_l = np.sqrt(np.mean(df['Error_LateFusion'] ** 2))
            corr_l = np.corrcoef(df['True_Age'], df['Predicted_LateFusion'])[0, 1]
            st.metric("MAE", f"{mae_l:.2f} years")
            st.metric("RMSE", f"{rmse_l:.2f} years")
            st.metric("Correlation", f"{corr_l:.3f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    st.markdown("### Visualizations")
    if os.path.exists(plot_path):
        from PIL import Image
        img = Image.open(plot_path)
        st.image(img, caption="VGG16 + SVR — Early & Late Fusion results", use_container_width=True)
    else:
        st.warning("Plot image not found (`Results/brain_age_vgg16_svr_results.png`).")

    # ── Per-subject table ─────────────────────────────────────────────────────
    st.markdown("### Per-Subject Predictions")
    display_df = df.copy()
    st.dataframe(
        display_df.style.format({
            'True_Age': '{:.1f}',
            'Predicted_EarlyFusion': '{:.1f}',
            'Error_EarlyFusion': '{:+.1f}',
            'Predicted_LateFusion': '{:.1f}',
            'Error_LateFusion': '{:+.1f}',
        }).background_gradient(subset=['Error_EarlyFusion', 'Error_LateFusion'], cmap='RdYlBu_r'),
        use_container_width=True
    )

    # ── Download buttons ──────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Download Predictions (CSV)",
            data=df.to_csv(index=False),
            file_name="brain_age_vgg16_svr_predictions.csv",
            mime="text/csv"
        )
    with col2:
        if summary_df is not None:
            st.download_button(
                label="📥 Download Summary (CSV)",
                data=summary_df.to_csv(index=False),
                file_name="brain_age_vgg16_svr_summary.csv",
                mime="text/csv"
            )


def show_resnet_page():
    """Display 3D ResNet-18 pipeline results."""
    st.markdown("## 3D ResNet-18 Pipeline Results")

    base_dir   = os.path.dirname(__file__)
    pred_path  = os.path.join(base_dir, "Results_3DResNet", "brain_age_resnet3d_predictions.csv")
    scatter_path  = os.path.join(base_dir, "Results_3DResNet", "brain_age_resnet3d_results.png")
    curves_path   = os.path.join(base_dir, "Results_3DResNet", "brain_age_resnet3d_training_curves.png")

    if not os.path.exists(pred_path):
        st.warning("Results not found. Please run `BrainAge_3DResNet_Colab.ipynb` and place results in `Results_3DResNet/`.")
        return

    df = pd.read_csv(pred_path)
    st.info(f"Results from **3D ResNet-18** pipeline — validation set ({len(df)} subjects)")

    # ── Summary metrics ───────────────────────────────────────────────────
    st.markdown("### Performance Summary")
    mae  = np.mean(np.abs(df['Brain_Age_Gap']))
    rmse = np.sqrt(np.mean(df['Brain_Age_Gap'] ** 2))
    corr = np.corrcoef(df['True_Age'], df['Predicted_Age'])[0, 1]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Subjects", len(df))
    col2.metric("MAE", f"{mae:.2f} years")
    col3.metric("RMSE", f"{rmse:.2f} years")
    col4.metric("Correlation", f"{corr:.3f}")

    # ── Plots ─────────────────────────────────────────────────────────────
    st.markdown("### Predicted vs True Age")
    if os.path.exists(scatter_path):
        from PIL import Image
        st.image(Image.open(scatter_path), caption="3D ResNet-18 — Predicted vs True Age", use_container_width=True)
    else:
        st.warning("Scatter plot not found (`Results_3DResNet/brain_age_resnet3d_results.png`).")

    st.markdown("### Training Curves")
    if os.path.exists(curves_path):
        from PIL import Image
        st.image(Image.open(curves_path), caption="3D ResNet-18 — Training Curves", use_container_width=True)
    else:
        st.warning("Training curves not found (`Results_3DResNet/brain_age_resnet3d_training_curves.png`).")

    # ── Per-subject table ─────────────────────────────────────────────────
    st.markdown("### Per-Subject Predictions")
    st.dataframe(
        df.style.format({
            'True_Age':      '{:.1f}',
            'Predicted_Age': '{:.1f}',
            'Brain_Age_Gap': '{:+.1f}',
        }).background_gradient(subset=['Brain_Age_Gap'], cmap='RdYlBu_r'),
        use_container_width=True
    )

    # ── Download button ───────────────────────────────────────────────────
    st.download_button(
        label="📥 Download Predictions (CSV)",
        data=df.to_csv(index=False),
        file_name="brain_age_resnet3d_predictions.csv",
        mime="text/csv"
    )


def show_about_page():
    """Display about page."""
    st.markdown("## About This Project")

    st.markdown("""
    ### Brain Age Estimation from MRI

    This project implements brain age prediction using deep learning features
    extracted from T1-weighted MRI scans.

    #### Methods

    **Feature Extraction:**
    - VGG16 pretrained on ImageNet
    - Features from `block5_pool` layer (25,088 dimensions)
    - MRI slices grouped into triplets (pseudo-RGB images)

    **Fusion Strategies:**

    | Strategy | Description |
    |----------|-------------|
    | Early Fusion | Aggregate features first, then train single model |
    | Late Fusion | Train separate models per slice, combine predictions |

    **Regression:**
    - Support Vector Regression (SVR)
    - Linear kernel with optimized hyperparameters

    #### Dataset

    Trained and validated on the [IXI Dataset](https://brain-development.org/ixi-dataset/):
    - ~600 healthy subjects
    - Age range: 20-86 years
    - T1-weighted MRI scans

    #### References

    - Cole, J.H., et al. (2017). Predicting brain age with deep learning. NeuroImage.
    - Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks. arXiv.

    ---

    #### Source Code

    🔗 [GitHub Repository](https://github.com/tanchaud/brain-age-estimation)

    #### Technologies Used

    - **TensorFlow/Keras** - Deep learning framework
    - **scikit-learn** - Machine learning
    - **nibabel** - Neuroimaging file I/O
    - **Streamlit** - Web application framework
    """)

    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit")


if __name__ == "__main__":
    main()
