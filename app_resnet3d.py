"""
Brain Age Estimation — 3D ResNet-18 Demo
=========================================
Streamlit web app for predicting brain age from T1 MRI scans
using the end-to-end 3D ResNet-18 model trained in Colab.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Age — 3D ResNet",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background: #f0f7ff;
        border-left: 4px solid #1E88E5;
        border-radius: 6px;
        padding: 1rem 1.5rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Model definition (must match the Colab training code exactly) ─────────────
@st.cache_resource
def load_resnet_model(checkpoint_path: str):
    """
    Loads the 3D ResNet-18 from a .pth checkpoint saved by the Colab notebook.
    Returns (model, device) on success, or (None, None) on failure.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class BasicBlock3D(nn.Module):
            def __init__(self, in_ch, out_ch, stride=1):
                super().__init__()
                self.conv1    = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
                self.bn1      = nn.BatchNorm3d(out_ch)
                self.conv2    = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
                self.bn2      = nn.BatchNorm3d(out_ch)
                self.shortcut = nn.Sequential()
                if stride != 1 or in_ch != out_ch:
                    self.shortcut = nn.Sequential(
                        nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                        nn.BatchNorm3d(out_ch),
                    )

            def forward(self, x):
                out = F.relu(self.bn1(self.conv1(x)), inplace=True)
                out = self.bn2(self.conv2(out))
                return F.relu(out + self.shortcut(x), inplace=True)

        class ResNet3D(nn.Module):
            def __init__(self, in_channels=1, dropout=0.3):
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv3d(in_channels, 64, 7, stride=2, padding=3, bias=False),
                    nn.BatchNorm3d(64), nn.ReLU(inplace=True),
                    nn.MaxPool3d(3, stride=2, padding=1),
                )
                self.layer1 = self._make(64,  64,  2, 1)
                self.layer2 = self._make(64,  128, 2, 2)
                self.layer3 = self._make(128, 256, 2, 2)
                self.layer4 = self._make(256, 512, 2, 2)
                self.pool    = nn.AdaptiveAvgPool3d(1)
                self.dropout = nn.Dropout(dropout)
                self.fc      = nn.Linear(512, 1)

            @staticmethod
            def _make(in_ch, out_ch, blocks, stride):
                layers = [BasicBlock3D(in_ch, out_ch, stride)]
                for _ in range(1, blocks):
                    layers.append(BasicBlock3D(out_ch, out_ch))
                return nn.Sequential(*layers)

            def forward(self, x):
                x = self.stem(x)
                x = self.layer1(x); x = self.layer2(x)
                x = self.layer3(x); x = self.layer4(x)
                return self.fc(self.dropout(self.pool(x).flatten(1))).squeeze(1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = ResNet3D(in_channels=1, dropout=0.3).to(device)

        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model, device, ckpt.get("val_mae"), ckpt.get("epoch")

    except Exception as exc:
        return None, None, None, str(exc)


def preprocess_nifti(tmp_path: str, img_size=(96, 96, 96)):
    """
    Applies the same MONAI preprocessing as the Colab val_transforms:
    RAS orientation → 2 mm isotropic → pad/crop to img_size → intensity normalise.
    Returns a float32 tensor [1, 1, D, H, W] ready for the model.
    """
    import torch
    import monai.transforms as T

    transforms = T.Compose([
        T.LoadImaged(keys=["image"]),
        T.EnsureChannelFirstd(keys=["image"]),
        T.Orientationd(keys=["image"], axcodes="RAS"),
        T.Spacingd(keys=["image"], pixdim=(2.0, 2.0, 2.0), mode="bilinear"),
        T.ResizeWithPadOrCropd(keys=["image"], spatial_size=img_size),
        T.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        T.ToTensord(keys=["image"]),
    ])
    item = transforms({"image": tmp_path})
    return item["image"].float().unsqueeze(0)   # [1, 1, D, H, W]


def plot_three_views(vol: np.ndarray, title: str = ""):
    """Show axial, coronal, and sagittal mid-slices of a 3D volume."""
    d, h, w = vol.shape
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    views = [
        (vol[d // 2, :, :], "Axial"),
        (vol[:, h // 2, :], "Coronal"),
        (vol[:, :, w // 2], "Sagittal"),
    ]
    for ax, (slice_, label) in zip(axes, views):
        ax.imshow(np.rot90(slice_), cmap="gray")
        ax.set_title(label, fontsize=12)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    return fig


def plot_predictions(df: pd.DataFrame):
    """Scatter + error-distribution plots for a results DataFrame."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    sc = ax.scatter(
        df["True_Age"], df["Predicted_Age"],
        c=df["Brain_Age_Gap"], cmap="RdYlBu_r",
        alpha=0.75, edgecolors="black", linewidths=0.4, s=65,
    )
    lo = min(df["True_Age"].min(), df["Predicted_Age"].min()) - 2
    hi = max(df["True_Age"].max(), df["Predicted_Age"].max()) + 2
    ax.plot([lo, hi], [lo, hi], "r--", lw=2, label="Perfect prediction")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Chronological Age (years)", fontsize=12)
    ax.set_ylabel("Predicted Brain Age (years)", fontsize=12)
    ax.set_title("Brain Age Prediction — 3D ResNet-18", fontsize=13)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label="Brain Age Gap (years)")

    ax = axes[1]
    errors = df["Brain_Age_Gap"]
    ax.hist(errors, bins=20, alpha=0.7, color="steelblue", edgecolor="black")
    ax.axvline(0,            color="red",    linestyle="--", lw=2, label="Zero error")
    ax.axvline(errors.mean(), color="orange", linestyle="-",  lw=2,
               label=f"Mean = {errors.mean():.2f}")
    ax.set_xlabel("Brain Age Gap (years)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Error Distribution", fontsize=13)
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def make_demo_data():
    """Synthetic results for the Demo Results page."""
    rng = np.random.default_rng(42)
    n   = 60
    true_ages = rng.uniform(20, 85, n)
    preds     = np.clip(true_ages + rng.normal(0, 4.8, n), 18, 92)
    return pd.DataFrame({
        "Subject_ID":    [f"IXI{i:03d}" for i in range(n)],
        "True_Age":      true_ages,
        "Predicted_Age": preds,
        "Brain_Age_Gap": preds - true_ages,
    })


# ── Pages ─────────────────────────────────────────────────────────────────────

def show_home():
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("## What is Brain Age?")
        st.markdown("""
        **Brain age** is a biomarker reflecting the biological age of the brain from neuroimaging.
        The gap between predicted brain age and chronological age (**Brain Age Gap, BAG**) can reveal:

        - 🧓 **Accelerated aging** — brain appears older than expected
        - 🧒 **Decelerated aging** — brain appears younger than expected
        - 🏥 **Potential neurological risk** — associated with disease or lifestyle factors
        """)

        st.markdown("## How This Model Works")
        st.markdown("""
        1. **Preprocessing** — MRI resampled to 2 mm isotropic, normalised to 96 × 96 × 96
        2. **3D ResNet-18** — full volume processed end-to-end (no slice-by-slice tricks)
        3. **Regression head** — single FC layer outputs predicted age directly
        4. **Brain Age Gap** — difference between prediction and chronological age
        """)

    with col2:
        st.markdown("## Model Summary")
        st.metric("Architecture", "3D ResNet-18")
        st.metric("Input size", "96 × 96 × 96")
        st.metric("Parameters", "~33 M")
        st.metric("Training loss", "L1 (MAE)")
        st.markdown("---")
        st.info("👈 Use the sidebar to navigate")


def show_upload(checkpoint_path: str):
    st.markdown("## Upload MRI Scan")

    # Dependency check
    missing = []
    for pkg in ("nibabel", "torch", "monai"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        st.error(f"Missing packages: {', '.join(missing)}")
        st.code(f"pip install {' '.join(missing)}", language="bash")
        return

    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        st.warning("No model checkpoint found. Enter the path to `best_resnet3d.pth` in the sidebar.")
        return

    # Load model
    with st.spinner("Loading 3D ResNet-18 checkpoint..."):
        model, device, val_mae, info = load_resnet_model(checkpoint_path)

    if model is None:
        st.error(f"Could not load model: {info}")
        return

    st.success(
        f"Model loaded on **{device}**"
        + (f" — best Val MAE: **{val_mae:.2f} yr** (epoch {info})" if val_mae else "")
    )

    # File upload
    uploaded = st.file_uploader(
        "Choose a NIfTI file (.nii or .nii.gz)",
        type=["nii", "gz"],
        help="T1-weighted MRI scan in NIfTI format",
    )

    col1, col2 = st.columns(2)
    with col1:
        true_age = st.number_input("Chronological age (optional)", min_value=0, max_value=120, value=0)
    with col2:
        st.info("Enter the subject's actual age to compute the Brain Age Gap")

    if uploaded is None:
        return

    suffix = ".nii.gz" if uploaded.name.endswith(".gz") else ".nii"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name

    try:
        import nibabel as nib
        nii  = nib.load(tmp_path)
        data = nii.get_fdata(dtype=np.float32)

        st.markdown("### Volume Information")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Dimensions", str(data.shape[:3]))
        c2.metric("Voxel size (mm)", str(tuple(round(v, 2) for v in nii.header.get_zooms()[:3])))
        c3.metric("Min intensity", f"{data.min():.1f}")
        c4.metric("Max intensity", f"{data.max():.1f}")

        st.markdown("### Mid-plane Views")
        vol3d = data if data.ndim == 3 else data[..., 0]
        st.pyplot(plot_three_views(vol3d))
        plt.close("all")

        if st.button("🔮 Predict Brain Age", type="primary"):
            with st.spinner("Preprocessing and running inference..."):
                import torch
                tensor = preprocess_nifti(tmp_path).to(device)
                with torch.no_grad():
                    predicted_age = model(tensor).item()

            st.markdown("### Results")
            r1, r2, r3 = st.columns(3)
            r1.metric("Predicted Brain Age", f"{predicted_age:.1f} yr")

            if true_age > 0:
                gap = predicted_age - true_age
                r2.metric("Chronological Age", f"{true_age} yr")
                r3.metric(
                    "Brain Age Gap",
                    f"{gap:+.1f} yr",
                    delta=f"{'Older' if gap > 0 else 'Younger'} than expected",
                )

                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                if abs(gap) <= 5:
                    st.success("✅ Brain age is within **±5 years** of chronological age — normal range.")
                elif gap > 5:
                    st.warning(
                        f"⚠️ Brain appears **{gap:.1f} years older** than chronological age. "
                        "Consider clinical follow-up."
                    )
                else:
                    st.info(
                        f"ℹ️ Brain appears **{abs(gap):.1f} years younger** than chronological age. "
                        "Generally a positive indicator."
                    )
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                r2.metric("Chronological Age", "Not provided")
                r3.metric("Brain Age Gap", "N/A")
                st.info("Enter chronological age above to compute the Brain Age Gap.")

    finally:
        os.unlink(tmp_path)


def show_demo():
    st.markdown("## Demo Results")

    csv_path = os.path.join(os.path.dirname(__file__), "brain_age_predictions.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if 'Error' in df.columns and 'Brain_Age_Gap' not in df.columns:
            df['Brain_Age_Gap'] = df['Error']
        st.markdown("Results loaded from the trained model.")
    else:
        st.warning("No real predictions found — showing synthetic demo data.")
        df = make_demo_data()

    mae  = np.mean(np.abs(df["Brain_Age_Gap"]))
    rmse = np.sqrt(np.mean(df["Brain_Age_Gap"] ** 2))
    corr = np.corrcoef(df["True_Age"], df["Predicted_Age"])[0, 1]

    st.markdown("### Model Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Subjects",    len(df))
    c2.metric("MAE",         f"{mae:.2f} yr")
    c3.metric("RMSE",        f"{rmse:.2f} yr")
    c4.metric("Pearson r",   f"{corr:.3f}")

    st.markdown("### Visualisation")
    st.pyplot(plot_predictions(df))
    plt.close("all")

    st.markdown("### Per-Subject Predictions")
    st.dataframe(
        df.style
          .format({"True_Age": "{:.1f}", "Predicted_Age": "{:.1f}", "Brain_Age_Gap": "{:+.1f}"})
          .background_gradient(subset=["Brain_Age_Gap"], cmap="RdYlBu_r"),
        use_container_width=True,
    )

    st.download_button(
        "📥 Download Results (CSV)",
        data=df.to_csv(index=False),
        file_name="demo_brain_age_resnet3d.csv",
        mime="text/csv",
    )


def show_about():
    st.markdown("## About This Project")
    st.markdown("""
    ### Brain Age Estimation — 3D ResNet-18

    This application predicts brain age from T1-weighted MRI scans using an
    **end-to-end 3D convolutional neural network**, replacing the earlier
    VGG16 feature extraction + SVR pipeline.

    #### Model Architecture

    | Component | Details |
    |-----------|---------|
    | Backbone | 3D ResNet-18 (4 residual stages) |
    | Input | 96 × 96 × 96, single channel |
    | Channels | 64 → 128 → 256 → 512 |
    | Head | AdaptiveAvgPool → Dropout(0.3) → FC(512 → 1) |
    | Parameters | ~33 million |

    #### Training Details

    | Setting | Value |
    |---------|-------|
    | Loss | L1 (MAE) |
    | Optimiser | Adam, lr = 1 × 10⁻⁴ |
    | LR schedule | Cosine annealing (60 epochs) |
    | Batch size | 2 (Colab T4 GPU) |
    | Precision | Mixed (AMP) |
    | Augmentation | Random flips, intensity scale/shift |

    #### Preprocessing (MONAI pipeline)

    1. Load NIfTI → reorient to RAS
    2. Resample to **2 mm isotropic** voxels
    3. Pad/crop to **96 × 96 × 96**
    4. Z-score normalise (non-zero voxels only)

    #### References

    - He, K., et al. (2016). Deep residual learning for image recognition. CVPR.
    - Cole, J.H., et al. (2017). Predicting brain age with deep learning. NeuroImage.
    - MONAI Consortium (2020). MONAI: Medical Open Network for AI. GitHub.

    ---
    **Technologies:** PyTorch · MONAI · nibabel · Streamlit
    """)
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.markdown('<p class="main-header">🧠 Brain Age Estimation</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">End-to-end 3D ResNet-18 · T1-weighted MRI → predicted age</p>',
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Upload MRI", "Demo Results", "About"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Checkpoint")
    checkpoint_path = st.sidebar.text_input(
        "Path to best_resnet3d.pth",
        value="best_resnet3d.pth",
        help="Local path to the .pth file saved by the Colab training notebook",
    )

    if page == "Home":
        show_home()
    elif page == "Upload MRI":
        show_upload(checkpoint_path)
    elif page == "Demo Results":
        show_demo()
    elif page == "About":
        show_about()


if __name__ == "__main__":
    main()
