"""
app.py  -  CLAM MIL Colon Cancer Detection - Streamlit Demo
============================================================
Run:
    streamlit run app.py
"""

import os
import glob
import tempfile
import time
from datetime import datetime

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import io
from PIL import Image as PILImage

from pipeline import run_full_inference, get_true_label
from report_generator import generate_pdf_report

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title  = "ColonAI - Cancer Detection",
    page_icon   = ":microscope:",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ============================================================================
# GLOBAL STYLE
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 18px 24px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
[data-testid="metric-container"] label {
    color: #a0aec0 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #e2e8f0 !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}

[data-testid="stFileUploader"] {
    border: 2px dashed #0f3460;
    border-radius: 12px;
    padding: 10px;
}

.result-card {
    border-radius: 14px;
    padding: 22px 28px;
    text-align: center;
    margin: 8px 0;
}
.tumor-card {
    background: linear-gradient(135deg, #4a0000 0%, #7b0000 100%);
    border: 1px solid #ff4444;
    box-shadow: 0 0 30px rgba(255,68,68,0.25);
}
.normal-card {
    background: linear-gradient(135deg, #003300 0%, #005500 100%);
    border: 1px solid #44ff44;
    box-shadow: 0 0 30px rgba(68,255,68,0.20);
}
.result-card h1 { margin: 0 0 6px 0; font-size: 2.4rem; }
.result-card p  { margin: 0; font-size: 0.95rem; color: #ccc; }

.section-heading {
    font-size: 1.15rem;
    font-weight: 600;
    color: #90cdf4;
    margin: 8px 0 4px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid #2d3748;
}
.info-box {
    background: #1a202c;
    border-left: 4px solid #4299e1;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    font-size: 0.9rem;
    color: #cbd5e0;
    line-height: 1.6;
}
.warn-box {
    background: #2d2000;
    border-left: 4px solid #f6ad55;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    font-size: 0.9rem;
    color: #fbd38d;
    line-height: 1.6;
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
    color: #e2e8f0;
    border: 1px solid #4299e1;
    border-radius: 10px;
    padding: 10px 24px;
    font-weight: 600;
    font-size: 0.95rem;
    transition: all 0.2s ease;
}
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #1a4a80 0%, #1a2a50 100%);
    border-color: #63b3ed;
    box-shadow: 0 0 16px rgba(66,153,225,0.35);
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS
# ============================================================================
MODEL_DIR    = os.path.join("models", "five_fold")
MODEL_PATHS  = sorted(glob.glob(os.path.join(MODEL_DIR, "best_fold_*.pt")))
THRESHOLD    = 0.276
DISPLAY_SIZE = 768

# ============================================================================
# HELPERS
# ============================================================================

def render_heatmap(grid: np.ndarray, title: str, figsize=(6, 5)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    im = ax.imshow(grid, cmap="jet", interpolation="bilinear",
                   origin="lower", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalised Attention", color="#a0aec0", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#a0aec0")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#a0aec0", fontsize=8)
    cbar.outline.set_edgecolor("#2d3748")
    ax.set_title(title, color="#e2e8f0", fontsize=12, fontweight="bold", pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d3748")
    plt.tight_layout(pad=0.5)
    return fig


def resize_for_display(image: np.ndarray, target_size: int = DISPLAY_SIZE) -> np.ndarray:
    """
    Resize image to fit within target_size × target_size while preserving aspect ratio.
    Pads with black to avoid layout shifts.
    """
    h, w = image.shape[:2]
    scale = target_size / float(max(h, w))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pad_top    = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left   = (target_size - new_w) // 2
    pad_right  = target_size - new_w - pad_left
    return cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )


def confidence_bar(prob: float, threshold: float = THRESHOLD):
    pct      = int(prob * 100)
    color    = "#e53e3e" if prob >= threshold else "#38a169"
    bg_color = "#1a202c"
    st.markdown(f"""
    <div style="margin: 6px 0 14px 0;">
      <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
        <span style="color:#a0aec0; font-size:0.8rem;">Normal (0%)</span>
        <span style="color:#a0aec0; font-size:0.8rem; font-weight:600;">
            Tumour probability: {pct}%
        </span>
        <span style="color:#a0aec0; font-size:0.8rem;">Tumour (100%)</span>
      </div>
      <div style="background:{bg_color}; border-radius:8px; height:14px; position:relative; border:1px solid #2d3748;">
        <div style="width:{pct}%; background:{color}; border-radius:8px;
                    height:100%; transition:width 0.4s ease;"></div>
        <div style="position:absolute; left:{int(threshold*100)}%;
                    top:0; bottom:0; width:2px; background:#f6ad55;"></div>
      </div>
      <div style="text-align:right; margin-top:3px;">
        <span style="color:#f6ad55; font-size:0.75rem;">
            &#9651; Decision threshold ({int(threshold*100)}%)
        </span>
      </div>
    </div>""", unsafe_allow_html=True)


def get_interpretation(grid: np.ndarray) -> str:
    idx   = np.unravel_index(np.argmax(grid), grid.shape)
    y, x  = idx
    rows, cols = grid.shape
    v_pos = "upper" if y > rows // 2 else "lower"
    h_pos = "right" if x > cols // 2 else "left"
    return (
        f"High attention observed in the {v_pos}-{h_pos} region, "
        "indicating possible tumour features in this area."
    )


def generate_report(result: dict, filename: str) -> bytes:
    """Thin wrapper — delegates to report_generator.generate_pdf_report()."""
    return generate_pdf_report(result, os.path.basename(filename))


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("## ColonAI")
    st.markdown("**CLAM - Attention MIL**")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    Analyses pre-extracted features from colon histopathology
    whole-slide images and classifies them as **Tumour** or **Normal**.

    Built with:
    - CLAM (attention-based MIL)
    - TCGA-COAD dataset
    - 5-fold cross-validation
    """)
    st.markdown("---")
    st.markdown("### Model Status")
    if MODEL_PATHS:
        st.success(f"OK - {len(MODEL_PATHS)} fold models loaded")
    else:
        st.error("No model checkpoints found")
    st.markdown("---")
    st.markdown("### Decision Threshold")
    st.info(f"Threshold = **{THRESHOLD}**\n\n(tuned to maximise F1)")
    st.markdown("---")
    st.markdown("### Heatmap Legend")
    st.markdown("""
    | Colour | Meaning |
    |--------|---------|
    | 🔴 Red / 🟡 Yellow | High attention |
    | 🟢 Green | Medium attention |
    | 🔵 Blue | Low attention |
    | ⚫ Black | No tissue / background |
    """)
    st.markdown("---")
    st.caption("Buddham Rajbhandari - Kaviya Darshini - Dakshini")


# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<div style="text-align:center; padding: 10px 0 6px 0;">
    <h1 style="font-size:2.4rem; font-weight:700; margin-bottom:4px;">
        ColonAI - Cancer Detection
    </h1>
    <p style="color:#90cdf4; font-size:1.1rem; margin:0; font-weight:500;">
        AI system for detecting colon cancer and highlighting suspicious regions
    </p>
    <p style="color:#a0aec0; font-size:0.9rem; margin-top:4px;">
        Attention-based Multiple Instance Learning on Histopathology WSIs
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")


# ============================================================================
# FILE UPLOAD
# ============================================================================
st.markdown('<p class="section-heading">Analysis Input</p>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
Upload a <code>.pt</code> file containing pre-extracted features for one WSI.<br>
Expected format: <code>{'features': tensor [N, 2048], 'coords': tensor [N, 2]}</code><br><br>
Optionally upload the original WSI thumbnail image (PNG/JPG/TIF) to enable the
<strong>overlay heatmap</strong> — the JET attention map blended onto your tissue image.
</div>
""", unsafe_allow_html=True)
st.markdown(" ")

col_up1, col_up2 = st.columns(2)
with col_up1:
    uploaded = st.file_uploader(
        "Feature bag (.pt)",
        type=["pt"],
        label_visibility="visible",
    )
with col_up2:
    uploaded_image = st.file_uploader(
        "Original WSI image (optional — enables overlay)",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        label_visibility="visible",
    )

# Controls
st.markdown(" ")
control_col1, control_col2, control_col3 = st.columns(3)
with control_col1:
    apply_smoothing = st.checkbox("Apply smoothing to heatmap", value=False,
                                  help="Light Gaussian blur on the attention map")
with control_col2:
    overlay_alpha = st.slider("Overlay opacity (alpha)", 0.3, 0.9, 0.6, step=0.05,
                              help="0.6 matches the reference target image")
with control_col3:
    sigma_value = None
    if apply_smoothing:
        sigma_value = st.slider("Gaussian sigma", 0.5, 10.0, 1.5, step=0.5)


# ============================================================================
# INFERENCE
# ============================================================================
if uploaded is not None:

    if not MODEL_PATHS:
        st.error("No trained model checkpoints found in models/five_fold/. Train the model first.")
        st.stop()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    # Load optional original image
    original_image_np = None
    if uploaded_image is not None:
        try:
            original_image_np = np.array(PILImage.open(uploaded_image).convert("RGB"))
            st.info(f"Original image loaded: {original_image_np.shape[1]}×{original_image_np.shape[0]} px")
        except Exception as e:
            st.error(f"Failed to read original image: {e}")
            os.remove(tmp_path)
            st.stop()

    with st.spinner("Running ensemble inference across all fold models…"):
        try:
            t0 = time.time()
            result = run_full_inference(
                tmp_path,
                MODEL_PATHS,
                original_image=original_image_np,
                patch_size=256,
                smoothing=apply_smoothing,
                smoothing_sigma=sigma_value,
                overlay_alpha=overlay_alpha,
            )
            inference_time = time.time() - t0
        except Exception as e:
            st.error(f"Inference failed: {e}")
            import traceback
            st.code(traceback.format_exc())
            os.remove(tmp_path)
            st.stop()

    os.remove(tmp_path)

    prob       = result["probability"]
    pred       = result["prediction"]
    label      = result["label"]
    confidence = result["confidence"]
    n_patches  = result["n_patches"]
    has_coords = result["has_coords"]
    true_label = get_true_label(uploaded.name)

    st.markdown("---")

    # =========================================================================
    # PREDICTION RESULT
    # =========================================================================
    st.markdown('<p class="section-heading">Slide-Level Prediction</p>', unsafe_allow_html=True)
    col_res, col_detail = st.columns([1, 1.6], gap="large")

    with col_res:
        if pred == 1:
            st.markdown("""
            <div class="result-card tumor-card">
                <h1>TUMOUR</h1>
                <p>Malignant tissue detected</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card normal-card">
                <h1>NORMAL</h1>
                <p>No malignancy detected</p>
            </div>""", unsafe_allow_html=True)

    with col_detail:
        st.markdown(f"**Uploaded Slide:** `{uploaded.name}`")
        st.markdown("**Tumour Probability**")
        confidence_bar(prob, THRESHOLD)
        d1, d2, d3, d4 = st.columns(4)
        with d1: st.metric("Probability %",    f"{prob*100:.1f}%")
        with d2: st.metric("Confidence",       f"{confidence:.3f}")
        with d3: st.metric("Patches",          f"{n_patches:,}")
        with d4: st.metric("Time",             f"{inference_time:.1f}s")
        if true_label is not None:
            correct    = (pred == true_label)
            icon       = "✅" if correct else "❌"
            label_name = "TUMOUR" if true_label == 1 else "NORMAL"
            st.markdown(f"**Ground Truth:** `{label_name}` ({icon})")

    st.markdown("---")

    # =========================================================================
    # ATTENTION HEATMAPS
    # =========================================================================
    if has_coords and result.get("heatmap_grid") is not None:
        st.markdown('<p class="section-heading">Spatial Attention Heatmaps</p>',
                    unsafe_allow_html=True)

        # Build tab list dynamically
        tab_labels = [
            "Coarse Grid (16×16)",
            "Spatial Heatmap",
        ]
        if result.get("overlay_image") is not None:
            tab_labels.append("Overlay on Original WSI")

        tabs = st.tabs(tab_labels)

        # ── Tab 0: Coarse Grid ────────────────────────────────────────────────
        with tabs[0]:
            st.markdown("""
            <div class="info-box" style="margin-bottom:12px;">
            <strong>Coarse 16×16 grid</strong> — each cell shows the mean attention of
            all patches that fell in that grid bin.
            Red/yellow = high attention (diagnostically significant).
            </div>""", unsafe_allow_html=True)

            fig_grid = render_heatmap(
                result["heatmap_grid"],
                "Coarse Attention Map (16×16)",
                figsize=(6, 5)
            )
            buf = io.BytesIO()
            fig_grid.savefig(buf, format="png", bbox_inches="tight", dpi=150)
            plt.close(fig_grid)
            buf.seek(0)
            img_grid = np.array(PILImage.open(buf).convert("RGB"))
            st.image(resize_for_display(img_grid, DISPLAY_SIZE), use_column_width=True)
            st.markdown(f"""
            <div class="info-box" style="margin-top:12px;">
            <strong>Interpretation:</strong> {get_interpretation(result["heatmap_grid"])}
            </div>""", unsafe_allow_html=True)

        # ── Tab 1: Standalone Spatial Heatmap ────────────────────────────────
        with tabs[1]:
            st.markdown("""
            <div class="info-box" style="margin-bottom:12px;">
            <strong>Per-patch JET heatmap</strong> — each 256×256 patch is individually
            coloured by its normalised attention score.
            Background (no tissue) = black.
            Red/yellow patches are the most diagnostically relevant.
            </div>""", unsafe_allow_html=True)

            if result.get("heatmap_spatial") is not None:
                spatial_display = resize_for_display(result["heatmap_spatial"], DISPLAY_SIZE)
                st.image(spatial_display, caption="Coordinate-based per-patch attention heatmap",
                         use_column_width=True)
            else:
                st.warning("Spatial heatmap not available for this slide.")

        # ── Tab 2: Overlay (only if original image was provided) ─────────────
        if result.get("overlay_image") is not None:
            with tabs[2]:
                st.markdown(f"""
                <div class="info-box" style="margin-bottom:12px;">
                <strong>Attention overlay</strong> blended onto original tissue image
                (alpha={overlay_alpha:.2f}).
                Formula: <code>result = {overlay_alpha:.2f} × JET(attention) + {1-overlay_alpha:.2f} × tissue</code><br>
                Background pixels = <code>JET(0) × {overlay_alpha:.2f}</code> ≈ dark navy blue.
                </div>""", unsafe_allow_html=True)

                overlay_display = resize_for_display(result["overlay_image"], DISPLAY_SIZE)
                st.image(overlay_display,
                         caption="JET attention heatmap overlaid on original WSI",
                         use_column_width=True)
        else:
            # Show a helpful note in a separate expander, not as a tab
            with st.expander("How to enable the Overlay tab"):
                st.markdown("""
                Upload your original WSI thumbnail image (PNG/JPG/TIF) using the
                **"Original WSI image"** uploader above, then re-run inference.
                The overlay blends the JET attention heatmap onto your tissue image
                at the selected alpha opacity.
                """)

        st.markdown("---")

    elif not has_coords:
        st.info("No patch coordinates found in this feature file — heatmaps are unavailable.")
        st.markdown("---")

    # =========================================================================
    # REPORT DOWNLOAD
    # =========================================================================
    st.markdown('<p class="section-heading">Diagnostic Report</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Download a structured PDF report containing the diagnostic summary,
    AI interpretation, all attention heatmaps, and methodology details.
    </div>
    """, unsafe_allow_html=True)
    st.markdown(" ")

    with st.spinner("Generating PDF report…"):
        try:
            pdf_bytes   = generate_report(result, uploaded.name)
            report_name = f"ColonAI_Report_{os.path.splitext(uploaded.name)[0]}.pdf"
            st.download_button(
                label               = "Download Diagnostic Report (PDF)",
                data                = pdf_bytes,
                file_name           = report_name,
                mime                = "application/pdf",
                use_container_width = True,
            )
            st.success("Report ready — click the button above to download.")
        except Exception as e:
            st.error(f"Report generation failed: {e}")

    st.markdown("---")


# ============================================================================
# MODEL LIMITATIONS
# ============================================================================
st.markdown('<p class="section-heading">Model Limitations and Known Failure Cases</p>',
            unsafe_allow_html=True)
st.markdown("""
<div class="warn-box">
<strong>What can go wrong?</strong><br><br>
<strong>False Negatives (missed tumours)</strong> — The model predicts <em>Normal</em>
for a slide that is actually tumour. In our 5-fold evaluation,
<strong>4 out of 23 tumour slides</strong> were missed (sensitivity = 81.8%).<br><br>
<strong>Under-confident predictions</strong> — Because training used only 46 balanced
slides, probabilities are systematically lower than 0.5 even for true tumours.
We compensate with a tuned threshold of <strong>0.276</strong>.<br><br>
<strong>Attention vs Pathologist annotation</strong> — High-attention regions are
statistically correlated with the slide label, but are <em>not</em> guaranteed to
correspond to tumour cells. Always confirm with expert review.
</div>
""", unsafe_allow_html=True)

st.markdown(" ")
with st.expander("Known False Negative Example from Cross-Validation"):
    col_fn1, col_fn2 = st.columns([1, 2])
    with col_fn1:
        st.markdown("""
        | Field | Value |
        |---|---|
        | **Slide** | TCGA-3L-AA1B-01Z |
        | **True label** | Tumour |
        | **Predicted** | Normal |
        | **Probability** | 0.196 |
        | **Threshold** | 0.276 |
        | **Patches** | 26,812 |
        """)
    with col_fn2:
        st.markdown("""
        <div class="warn-box">
        This tumour slide scored only <strong>0.196</strong> — well below
        the 0.276 threshold — and was classified as Normal.
        The attention heatmap shows diffuse low-level activation with no
        strong focal hotspot, making it a challenging case.
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# HOW IT WORKS
# ============================================================================
with st.expander("How the Pipeline Works"):
    st.markdown("""
    ```
    WSI (.svs)
       -> Tissue segmentation + patch extraction (256×256 px at 20×)
       -> ResNet-50 feature extraction  →  [N, 2048] per patch
       -> Saved as .pt file { features, coords }
          ---- your upload starts here ----
       -> AttentionNet: Linear(2048→512) → Tanh → Dropout → Linear(512→1)
       -> Softmax over ALL N patches  →  attention distribution [N]
       -> Weighted sum  →  slide representation [2048]
       -> Classifier: Linear(2048→1) + Sigmoid  →  probability
       -> 5× fold ensemble (average probabilities + attention)
       -> Threshold @ 0.276  →  Tumour / Normal
       -> Grid reconstruction (16×16)  →  coarse grid heatmap
       -> Per-patch JET stamp  →  spatial heatmap + overlay
    ```
    **Overlay formula:** `result = alpha × JET(attention) + (1 − alpha) × tissue`
    Background pixels (no tissue): `alpha × JET(0)` ≈ dark navy blue at alpha=0.6
    """)

st.markdown("---")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<div style="text-align:center; color:#4a5568; font-size:0.8rem; padding: 8px 0 20px 0;">
    ColonAI — CLAM-based MIL for Colon Cancer Detection —
    Buddham Rajbhandari &nbsp;·&nbsp; Kaviya Darshini &nbsp;·&nbsp; Dakshini
    <br>
    <span style="font-size:0.72rem;">
    For research and demonstration purposes only.
    Not a substitute for expert pathological diagnosis.
    </span>
</div>
""", unsafe_allow_html=True)
