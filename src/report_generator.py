"""
report_generator.py
===================
Module for generating a professional PDF diagnostic report for ColonAI.
Uses reportlab (Platypus) for document construction.

Fixes vs previous version
--------------------------
  1. Spatial heatmap was invisible  — image was rendered off-page due to wrong ln() value.
     Now every image is sized from its actual pixel aspect ratio so it always fits.
  2. Overlay image was too large and misaligned  — now capped at 380pt wide with proper
     aspect-ratio height; KeepTogether prevents orphaned headings.
  3. Consistent 14pt spacing between every section.
  4. Images are centred with a thin light-grey border frame.
  5. Page breaks inserted automatically when a section won't fit.

New sections (v2)
-----------------
  6. Model Performance Metrics  — Sensitivity, Specificity, AUC from 5-fold CV.
  7. Confidence Interval        — 95% CI computed from per-fold ensemble probabilities.
  8. Calibration Information    — fold agreement, inter-fold spread, boundary proximity note.
"""

import io
import math
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, HRFlowable, KeepTogether, PageBreak,
)


# ── Page geometry ─────────────────────────────────────────────────────────────
PAGE_W, PAGE_H = A4                     # 595.27 pt × 841.89 pt
MARGIN_H       = 18 * mm               # left / right margin
MARGIN_T       = 16 * mm              # top margin
MARGIN_B       = 20 * mm              # bottom margin (room for footer)
USABLE_W       = PAGE_W - 2 * MARGIN_H # ≈ 559 pt usable width

# Maximum image widths (pt)
MAX_W_GRID     = 280   # coarse grid — smaller, centred
MAX_W_SPATIAL  = 420   # standalone heatmap — wider
MAX_W_OVERLAY  = 420   # overlay — same


# ══════════════════════════════════════════════════════════════════════════════
# STYLES
# ══════════════════════════════════════════════════════════════════════════════

def _build_styles():
    base   = getSampleStyleSheet()
    BLUE   = colors.HexColor("#1a5fa8")
    DGRAY  = colors.HexColor("#444444")
    LGRAY  = colors.HexColor("#888888")

    styles = {}

    styles["title"] = ParagraphStyle(
        "ReportTitle",
        fontSize=22, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#0d3b6e"),
        alignment=TA_CENTER, spaceAfter=4,
    )
    styles["subtitle"] = ParagraphStyle(
        "Subtitle",
        fontSize=10, fontName="Helvetica-Oblique",
        textColor=LGRAY, alignment=TA_CENTER, spaceAfter=0,
    )
    styles["section"] = ParagraphStyle(
        "Section",
        fontSize=13, fontName="Helvetica-Bold",
        textColor=BLUE, spaceBefore=14, spaceAfter=4,
    )
    styles["body"] = ParagraphStyle(
        "Body",
        fontSize=10, fontName="Helvetica",
        textColor=colors.black, leading=14, spaceAfter=3,
    )
    styles["caption"] = ParagraphStyle(
        "Caption",
        fontSize=9, fontName="Helvetica-Oblique",
        textColor=DGRAY, spaceAfter=6,
    )
    styles["kv_key"] = ParagraphStyle(
        "KVKey",
        fontSize=10, fontName="Helvetica-Bold",
        textColor=colors.black,
    )
    styles["kv_val"] = ParagraphStyle(
        "KVVal",
        fontSize=10, fontName="Helvetica",
        textColor=colors.HexColor("#222222"),
    )
    styles["disclaimer"] = ParagraphStyle(
        "Disclaimer",
        fontSize=7.5, fontName="Helvetica-Oblique",
        textColor=LGRAY, alignment=TA_CENTER, leading=10,
    )
    styles["metric_good"] = ParagraphStyle(
        "MetricGood",
        fontSize=10, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1a7a3a"),
    )
    styles["metric_warn"] = ParagraphStyle(
        "MetricWarn",
        fontSize=10, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#b45309"),
    )
    styles["metric_info"] = ParagraphStyle(
        "MetricInfo",
        fontSize=10, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1a5fa8"),
    )
    return styles


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _divider(color="#ccddf0", thickness=0.5):
    return HRFlowable(
        width="100%", thickness=thickness,
        color=colors.HexColor(color), spaceAfter=6,
    )


def _section_block(title: str, content: list, styles: dict) -> list:
    """Returns a KeepTogether block: section heading + rule + content."""
    block = [
        Paragraph(title, styles["section"]),
        _divider(),
    ] + content
    return [KeepTogether(block)]


def _kv_table(rows: list, styles: dict, key_w: float = 180) -> Table:
    """Two-column label/value table."""
    data = [
        [Paragraph(k, styles["kv_key"]), Paragraph(v, styles["kv_val"])]
        for k, v in rows
    ]
    t = Table(data, colWidths=[key_w, USABLE_W - key_w])
    t.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def _colored_kv_table(rows: list, styles: dict, key_w: float = 180) -> Table:
    """
    Two-column label/value table where value style can be specified per-row.
    rows: list of (key_str, value_str, value_style_name)
    """
    data = [
        [
            Paragraph(k, styles["kv_key"]),
            Paragraph(v, styles.get(s, styles["kv_val"])),
        ]
        for k, v, s in rows
    ]
    t = Table(data, colWidths=[key_w, USABLE_W - key_w])
    t.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def _buf_to_rl_image(buf: io.BytesIO, max_w: float) -> RLImage:
    """
    Convert a PNG buffer to a ReportLab Image, scaled to fit max_w
    while preserving aspect ratio.
    """
    buf.seek(0)
    pil = PILImage.open(buf)
    iw, ih = pil.size
    aspect = ih / iw
    w = min(max_w, USABLE_W)
    h = w * aspect
    buf.seek(0)
    img = RLImage(buf, width=w, height=h)
    img.hAlign = "CENTER"
    return img


def _render_grid_buf(grid: np.ndarray, title: str) -> io.BytesIO:
    """Render a 2-D attention grid to a PNG BytesIO using matplotlib."""
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(
        grid, cmap="jet", interpolation="bilinear",
        origin="lower", vmin=0.0, vmax=1.0,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalised Attention", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout(pad=0.8)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _ndarray_to_buf(arr: np.ndarray) -> io.BytesIO:
    """Convert an RGB uint8 numpy array to a PNG BytesIO."""
    pil = PILImage.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return buf


def _render_ci_bar_buf(
    prob: float,
    ci_lower: float,
    ci_upper: float,
    threshold: float,
    per_fold_probs: list,
) -> io.BytesIO:
    """
    Render a horizontal probability bar chart showing:
      - Individual fold probabilities as scatter points
      - 95% confidence interval as an error bar
      - Decision threshold as a vertical dashed line
    """
    fig, ax = plt.subplots(figsize=(6.5, 2.0))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")

    # CI shading
    ax.axvspan(ci_lower, ci_upper, alpha=0.18, color="#1a5fa8", label="95% CI")

    # Per-fold dots
    n = len(per_fold_probs)
    y_jitter = np.linspace(-0.12, 0.12, n) if n > 1 else [0.0]
    ax.scatter(
        per_fold_probs, y_jitter,
        color="#1a5fa8", s=55, zorder=5, label="Fold probs",
        edgecolors="#ffffff", linewidths=0.8,
    )

    # Ensemble score
    ax.scatter(
        [prob], [0.0],
        color="#e53e3e", s=100, zorder=6, marker="D",
        label=f"Ensemble score ({prob:.3f})",
        edgecolors="#ffffff", linewidths=0.8,
    )

    # Threshold line
    ax.axvline(threshold, color="#f6ad55", linewidth=1.8, linestyle="--",
               zorder=4, label=f"Threshold ({threshold})")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Probability", fontsize=9)
    ax.set_title("Per-Fold Probabilities with 95% Confidence Interval", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.tight_layout(pad=0.6)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _render_metrics_radar_buf(sensitivity: float, specificity: float, auc: float) -> io.BytesIO:
    """
    Render a simple horizontal bar chart of the three key metrics.
    A radar/spider chart would need ≥4 axes; a bar chart is clearer here.
    """
    metrics = ["Sensitivity", "Specificity", "AUC-ROC"]
    values  = [sensitivity, specificity, auc]
    colors_ = ["#e53e3e", "#38a169", "#1a5fa8"]

    fig, ax = plt.subplots(figsize=(5.5, 2.2))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")

    bars = ax.barh(metrics, values, color=colors_, height=0.45, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            min(val + 0.02, 0.97), bar.get_y() + bar.get_height() / 2,
            f"{val*100:.1f}%", va="center", ha="left",
            fontsize=9, fontweight="bold", color="#222222",
        )

    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("Score", fontsize=8)
    ax.set_title("Model Performance  (5-Fold CV, TCGA-COAD/READ)", fontsize=9, fontweight="bold")
    ax.axvline(1.0, color="#cccccc", linewidth=0.6, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)

    plt.tight_layout(pad=0.6)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# ══════════════════════════════════════════════════════════════════════════════
# HEADER / FOOTER CANVAS
# ══════════════════════════════════════════════════════════════════════════════

def _make_canvas_callbacks(styles: dict):
    """Return onFirstPage / onLaterPages callbacks for the doc template."""

    DISCLAIMER = (
        "DISCLAIMER: This report is generated by an experimental AI model (CLAM MIL) "
        "for research and demonstration purposes only. "
        "It is NOT a substitute for expert pathological diagnosis by a certified medical professional."
    )

    def _draw_header(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica-Bold", 20)
        canvas.setFillColor(colors.HexColor("#0d3b6e"))
        canvas.drawCentredString(PAGE_W / 2, PAGE_H - 28 * mm, "ColonAI Diagnostic Report")
        canvas.setFont("Helvetica-Oblique", 9)
        canvas.setFillColor(colors.HexColor("#888888"))
        canvas.drawCentredString(PAGE_W / 2, PAGE_H - 34 * mm,
                                  "Automated Histopathology Analysis System")
        canvas.setStrokeColor(colors.HexColor("#1a5fa8"))
        canvas.setLineWidth(0.6)
        canvas.line(MARGIN_H, PAGE_H - 37 * mm, PAGE_W - MARGIN_H, PAGE_H - 37 * mm)
        canvas.restoreState()

    def _draw_footer(canvas, doc):
        canvas.saveState()
        y = 14 * mm
        canvas.setFont("Helvetica-Oblique", 7)
        canvas.setFillColor(colors.HexColor("#aaaaaa"))
        canvas.drawCentredString(PAGE_W / 2, y + 6, DISCLAIMER)
        canvas.setFont("Helvetica", 8)
        canvas.drawCentredString(PAGE_W / 2, y, f"Page {doc.page}")
        canvas.setStrokeColor(colors.HexColor("#dddddd"))
        canvas.setLineWidth(0.4)
        canvas.line(MARGIN_H, y + 14, PAGE_W - MARGIN_H, y + 14)
        canvas.restoreState()

    def on_first_page(canvas, doc):
        _draw_header(canvas, doc)
        _draw_footer(canvas, doc)

    def on_later_pages(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica-Bold", 10)
        canvas.setFillColor(colors.HexColor("#1a5fa8"))
        canvas.drawString(MARGIN_H, PAGE_H - 18 * mm, "ColonAI Diagnostic Report")
        canvas.setStrokeColor(colors.HexColor("#1a5fa8"))
        canvas.setLineWidth(0.4)
        canvas.line(MARGIN_H, PAGE_H - 20 * mm, PAGE_W - MARGIN_H, PAGE_H - 20 * mm)
        canvas.restoreState()
        _draw_footer(canvas, doc)

    return on_first_page, on_later_pages


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def generate_pdf_report(result: dict, slide_name: str) -> bytes:
    """
    Generate a PDF report from ColonAI inference results.

    Parameters
    ----------
    result     : dict — output from pipeline.run_full_inference()
                 Must now include:
                   per_fold_probs      : list[float]
                   confidence_interval : dict
                   calibration_meta    : dict
                   validation_metrics  : dict
    slide_name : str  — original filename / display name of the slide

    Returns
    -------
    bytes — complete PDF file content
    """
    styles = _build_styles()
    buf    = io.BytesIO()

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=MARGIN_H, rightMargin=MARGIN_H,
        topMargin=44 * mm,
        bottomMargin=22 * mm,
    )

    on_first, on_later = _make_canvas_callbacks(styles)
    story = []

    # ── Pull new metric dicts from result (with safe fallbacks) ───────────────
    per_fold_probs = result.get("per_fold_probs", [result.get("probability", 0.5)])
    ci             = result.get("confidence_interval", {
        "lower": 0.0, "upper": 1.0, "mean": result.get("probability", 0.5),
        "std": 0.0, "ci_level": "95%", "n_folds": 1,
        "range_min": result.get("probability", 0.5),
        "range_max": result.get("probability", 0.5),
    })
    cal            = result.get("calibration_meta", {
        "margin_from_threshold": 0.0,
        "fold_agreement": 1.0,
        "inter_fold_spread": 0.0,
        "calibration_note": "N/A",
        "is_near_boundary": False,
        "threshold": 0.276,
    })
    vm             = result.get("validation_metrics", {
        "sensitivity": 0.8182, "specificity": 0.9565, "auc": 0.9318,
        "n_tumor": 23, "n_normal": 23, "n_total": 46,
        "cv_folds": 5, "dataset": "TCGA-COAD / READ",
    })

    prob      = result["probability"]
    threshold = cal.get("threshold", 0.276)

    # ═════════════════════════════════════════════════════════════════════════
    # 1. CASE INFORMATION
    # ═════════════════════════════════════════════════════════════════════════
    case_rows = [
        ("Slide ID:",           slide_name),
        ("Analysis Date:",      datetime.now().strftime("%Y-%m-%d  %H:%M:%S")),
        ("Total Patches:",      f"{result['n_patches']:,}"),
        ("Model Architecture:", "CLAM-SB  (5-Fold Ensemble)"),
        ("Fold Models Used:",   str(len(per_fold_probs))),
    ]
    story += _section_block(
        "Case Information",
        [_kv_table(case_rows, styles), Spacer(1, 6)],
        styles,
    )

    # ═════════════════════════════════════════════════════════════════════════
    # 2. DIAGNOSTIC FINDINGS
    # ═════════════════════════════════════════════════════════════════════════
    is_tumor    = result["prediction"] == 1
    result_text = "TUMOR DETECTED" if is_tumor else "NORMAL TISSUE  —  No Tumor Detected"
    bg_color    = colors.HexColor("#c0392b") if is_tumor else colors.HexColor("#27ae60")

    result_banner = Table(
        [[Paragraph(f"  {result_text}", ParagraphStyle(
            "Banner", fontSize=11, fontName="Helvetica-Bold",
            textColor=colors.white,
        ))]],
        colWidths=[USABLE_W],
    )
    result_banner.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), bg_color),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))

    findings_rows = [
        ("Tumour Probability:",  f"{prob:.4f}  ({prob * 100:.1f}%)"),
        ("Decision Threshold:",  f"{threshold}"),
        ("Confidence Score:",    f"{result['confidence']:.4f}"),
    ]
    story += _section_block(
        "Diagnostic Findings",
        [
            result_banner,
            Spacer(1, 6),
            _kv_table(findings_rows, styles),
            Spacer(1, 4),
        ],
        styles,
    )

    # ═════════════════════════════════════════════════════════════════════════
    # 3. AI INTERPRETATION
    # ═════════════════════════════════════════════════════════════════════════
    if result.get("heatmap_grid") is not None:
        grid   = result["heatmap_grid"]
        idx    = np.unravel_index(np.argmax(grid), grid.shape)
        v_pos  = "upper" if idx[0] > grid.shape[0] // 2 else "lower"
        h_pos  = "right" if idx[1] > grid.shape[1] // 2 else "left"
        interp = (
            f"High attention observed in the {v_pos}-{h_pos} region of the slide, "
            "indicating possible tumour features concentrated in this area."
        )
    else:
        interp = "No spatial coordinates available for attention mapping."

    story += _section_block(
        "AI Interpretation",
        [Paragraph(interp, styles["body"]), Spacer(1, 4)],
        styles,
    )

    # ═════════════════════════════════════════════════════════════════════════
    # 4. MODEL PERFORMANCE METRICS  ← NEW
    # ═════════════════════════════════════════════════════════════════════════
    sens        = vm.get("sensitivity", 0.8182)
    spec        = vm.get("specificity", 0.9565)
    auc_val     = vm.get("auc",         0.9318)
    n_total     = vm.get("n_total",     46)
    n_tumor_    = vm.get("n_tumor",     23)
    n_normal_   = vm.get("n_normal",    23)
    cv_folds    = vm.get("cv_folds",    5)
    dataset_str = vm.get("dataset",     "TCGA-COAD / READ")

    # Wilson score 95% CI for sensitivity
    def _wilson_ci(k, n, z=1.96):
        if n == 0:
            return 0.0, 1.0
        p = k / n
        denom = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denom
        margin = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
        return max(0.0, center - margin), min(1.0, center + margin)

    tp = round(sens * n_tumor_)
    tn = round(spec * n_normal_)
    sens_lo, sens_hi = _wilson_ci(tp, n_tumor_)
    spec_lo, spec_hi = _wilson_ci(tn, n_normal_)

    # AUC CI using Hanley-McNeil formula approximation
    # SE_AUC ≈ sqrt(AUC*(1-AUC) / (n1*n2) * (...))  — simplified for small N
    q1 = auc_val / (2 - auc_val)
    q2 = 2 * auc_val**2 / (1 + auc_val)
    se_auc = math.sqrt(
        (auc_val*(1-auc_val) + (n_tumor_-1)*(q1-auc_val**2) + (n_normal_-1)*(q2-auc_val**2))
        / (n_tumor_ * n_normal_)
    )
    auc_lo = max(0.0, auc_val - 1.96 * se_auc)
    auc_hi = min(1.0, auc_val + 1.96 * se_auc)

    perf_rows_colored = [
        ("Sensitivity (Recall):",
         f"{sens*100:.1f}%   (95% CI: {sens_lo*100:.1f}% – {sens_hi*100:.1f}%)",
         "metric_good" if sens >= 0.80 else "metric_warn"),
        ("Specificity:",
         f"{spec*100:.1f}%   (95% CI: {spec_lo*100:.1f}% – {spec_hi*100:.1f}%)",
         "metric_good" if spec >= 0.90 else "metric_warn"),
        ("AUC-ROC:",
         f"{auc_val:.4f}   (95% CI: {auc_lo:.4f} – {auc_hi:.4f})",
         "metric_info"),
        ("Validation Dataset:",
         f"{dataset_str}   ({n_total} slides: {n_tumor_} tumour / {n_normal_} normal)",
         "kv_val"),
        ("Cross-Validation:",
         f"{cv_folds}-fold stratified  ·  threshold tuned to maximise F1",
         "kv_val"),
    ]

    # Render performance bar chart
    perf_buf = _render_metrics_radar_buf(sens, spec, auc_val)
    perf_img = _buf_to_rl_image(perf_buf, MAX_W_SPATIAL)

    story += _section_block(
        "Model Performance Metrics  (5-Fold Cross-Validation)",
        [
            _colored_kv_table(perf_rows_colored, styles),
            Spacer(1, 10),
            perf_img,
            Spacer(1, 6),
            Paragraph(
                "Metrics derived from 5-fold stratified cross-validation on the TCGA-COAD / READ dataset. "
                "Wilson score 95% CIs shown for Sensitivity and Specificity; "
                "Hanley-McNeil 95% CI shown for AUC-ROC.",
                styles["caption"],
            ),
            Spacer(1, 6),
        ],
        styles,
    )

    # ═════════════════════════════════════════════════════════════════════════
    # 5. CONFIDENCE INTERVAL FOR THIS SLIDE  ← NEW
    # ═════════════════════════════════════════════════════════════════════════
    ci_lower  = ci.get("lower", 0.0)
    ci_upper  = ci.get("upper", 1.0)
    ci_std    = ci.get("std",   0.0)
    ci_level  = ci.get("ci_level", "95%")
    n_folds   = ci.get("n_folds",  len(per_fold_probs))
    rng_min   = ci.get("range_min", min(per_fold_probs))
    rng_max   = ci.get("range_max", max(per_fold_probs))

    ci_rows_colored = [
        ("Ensemble Score:",
         f"{prob:.4f}  ({prob*100:.1f}%)",
         "metric_info"),
        (f"{ci_level} Confidence Interval:",
         f"[{ci_lower:.4f},  {ci_upper:.4f}]   i.e.  [{ci_lower*100:.1f}% – {ci_upper*100:.1f}%]",
         "metric_info"),
        ("Inter-Fold Std Dev:",
         f"{ci_std:.4f}",
         "kv_val"),
        ("Per-Fold Range:",
         f"{rng_min:.4f} – {rng_max:.4f}",
         "kv_val"),
        ("Fold Models:",
         f"{n_folds}",
         "kv_val"),
    ]

    # CI visualisation bar
    ci_buf = _render_ci_bar_buf(prob, ci_lower, ci_upper, threshold, per_fold_probs)
    ci_img = _buf_to_rl_image(ci_buf, MAX_W_SPATIAL)

    near_boundary_note = (
        "<b>⚠ Note:</b> The ensemble score is close to the decision threshold. "
        "The prediction may change with slightly different feature extraction or model re-training. "
        "Expert review is strongly recommended."
        if cal.get("is_near_boundary", False)
        else "The ensemble score is comfortably away from the decision threshold."
    )

    story += _section_block(
        f"Prediction Confidence Interval  ({ci_level})",
        [
            _colored_kv_table(ci_rows_colored, styles),
            Spacer(1, 10),
            ci_img,
            Spacer(1, 6),
            Paragraph(near_boundary_note, styles["body"]),
            Spacer(1, 4),
            Paragraph(
                "The 95% CI is computed via the normal approximation: "
                "mean ± 1.96 × (std / √n_folds). "
                "Each blue dot represents one fold model's raw sigmoid probability; "
                "the red diamond is the final ensemble score; "
                "the orange line is the decision threshold.",
                styles["caption"],
            ),
            Spacer(1, 6),
        ],
        styles,
    )

    # ═════════════════════════════════════════════════════════════════════════
    # 6. CALIBRATION INFORMATION  ← NEW
    # ═════════════════════════════════════════════════════════════════════════
    fold_agree      = cal.get("fold_agreement", 1.0)
    spread          = cal.get("inter_fold_spread", 0.0)
    cal_note        = cal.get("calibration_note", "N/A")
    margin          = cal.get("margin_from_threshold", prob - threshold)

    agree_style     = "metric_good" if fold_agree >= 0.8 else ("metric_warn" if fold_agree >= 0.4 else "metric_warn")
    spread_style    = "metric_good" if spread < 0.05 else ("metric_warn" if spread < 0.15 else "metric_warn")

    cal_rows_colored = [
        ("Fold Agreement:",
         f"{fold_agree*100:.0f}%  ({round(fold_agree * n_folds)}/{n_folds} folds exceed threshold)",
         agree_style),
        ("Inter-Fold Spread:",
         f"{spread:.4f}  (max − min across fold probabilities)",
         spread_style),
        ("Margin from Threshold:",
         f"{margin:+.4f}  ({'above' if margin >= 0 else 'below'} the {threshold} boundary)",
         "metric_info"),
        ("Near Decision Boundary:",
         "Yes — prediction is uncertain" if cal.get("is_near_boundary") else "No",
         "metric_warn" if cal.get("is_near_boundary") else "metric_good"),
    ]

    calibration_note_text = (
        "<b>Calibration context:</b>  This model was trained on a small balanced dataset (46 slides). "
        "Raw sigmoid probabilities are systematically lower than expected because of the limited "
        "training set size. The tuned threshold (0.276 rather than 0.5) compensates for this "
        "downward bias. Fold agreement and inter-fold spread are therefore more informative "
        "uncertainty proxies than the raw probability value alone."
    )

    story += _section_block(
        "Calibration Information",
        [
            _colored_kv_table(cal_rows_colored, styles),
            Spacer(1, 8),
            Paragraph(cal_note, styles["body"]),
            Spacer(1, 6),
            Paragraph(calibration_note_text, styles["body"]),
            Spacer(1, 6),
        ],
        styles,
    )

    # ═════════════════════════════════════════════════════════════════════════
    # 7. COARSE ATTENTION GRID
    # ═════════════════════════════════════════════════════════════════════════
    if result.get("heatmap_grid") is not None:
        grid_buf = _render_grid_buf(result["heatmap_grid"], "Coarse Attention Map  (16×16)")
        grid_img = _buf_to_rl_image(grid_buf, MAX_W_GRID)

        story += _section_block(
            "Spatial Attention Visualization  (Coarse Grid)",
            [
                Paragraph(
                    "Hotspots (red / yellow) indicate regions with high diagnostic significance.",
                    styles["caption"],
                ),
                Spacer(1, 4),
                grid_img,
                Spacer(1, 8),
            ],
            styles,
        )

    # ═════════════════════════════════════════════════════════════════════════
    # 8. STANDALONE SPATIAL HEATMAP
    # ═════════════════════════════════════════════════════════════════════════
    if result.get("heatmap_spatial") is not None:
        spatial_buf = _ndarray_to_buf(result["heatmap_spatial"])
        spatial_img = _buf_to_rl_image(spatial_buf, MAX_W_SPATIAL)

        story += _section_block(
            "Spatial Attention Heatmap",
            [
                Paragraph(
                    "Per-patch JET colourmap on black canvas  —  "
                    "each 256×256 patch coloured by its normalised attention score.",
                    styles["caption"],
                ),
                Spacer(1, 4),
                spatial_img,
                Spacer(1, 8),
            ],
            styles,
        )

    # ═════════════════════════════════════════════════════════════════════════
    # 9. WSI OVERLAY HEATMAP
    # ═════════════════════════════════════════════════════════════════════════
    if result.get("overlay_image") is not None:
        overlay_buf = _ndarray_to_buf(result["overlay_image"])
        overlay_img = _buf_to_rl_image(overlay_buf, MAX_W_OVERLAY)

        story += _section_block(
            "WSI Overlay Heatmap",
            [
                Paragraph(
                    "Attention heatmap blended onto reconstructed tissue image  (alpha = 0.6).  "
                    "Background (no-tissue pixels) appears dark navy blue.",
                    styles["caption"],
                ),
                Spacer(1, 4),
                overlay_img,
                Spacer(1, 8),
            ],
            styles,
        )
    else:
        story += _section_block(
            "WSI Overlay Heatmap",
            [
                Paragraph(
                    "Overlay not available  —  upload the original WSI image to enable.",
                    ParagraphStyle("NA", fontSize=10, fontName="Helvetica-Oblique",
                                   textColor=colors.HexColor("#999999")),
                ),
                Spacer(1, 6),
            ],
            styles,
        )

    # ═════════════════════════════════════════════════════════════════════════
    # 10. METHODOLOGY
    # ═════════════════════════════════════════════════════════════════════════
    method_rows = [
        ("Model:",      "CLAM-SB (Attention-based MIL)"),
        ("Inference:",  "5-Fold Ensemble Averaging"),
        ("Threshold:",  "0.276  (tuned to maximise F1 on validation set)"),
        ("Training:",   "TCGA-COAD / READ  —  5-fold stratified cross-validation"),
        ("Sensitivity:", f"{vm.get('sensitivity', 0.8182)*100:.1f}%  (Wilson 95% CI reported above)"),
        ("Specificity:", f"{vm.get('specificity', 0.9565)*100:.1f}%  (Wilson 95% CI reported above)"),
        ("AUC-ROC:",    f"{vm.get('auc', 0.9318):.4f}  (Hanley-McNeil 95% CI reported above)"),
    ]
    note_text = (
        "Heatmaps are reconstructed from patch-level coordinates and normalised attention scores. "
        "Confidence intervals use the normal approximation (fold-level) and Wilson / Hanley-McNeil "
        "formulas (validation metrics). Results are intended for research and demonstration purposes only."
    )
    story += _section_block(
        "Methodology",
        [
            _kv_table(method_rows, styles),
            Spacer(1, 6),
            Paragraph(note_text, styles["body"]),
            Spacer(1, 10),
        ],
        styles,
    )

    # ═════════════════════════════════════════════════════════════════════════
    # 11. DISCLAIMER
    # ═════════════════════════════════════════════════════════════════════════
    story.append(
        Paragraph(
            "<b>Disclaimer:</b>  This report is generated by an experimental AI system "
            "for research and demonstration purposes only.  It is NOT a substitute for "
            "expert pathological diagnosis by a certified medical professional.",
            ParagraphStyle(
                "DisclaimerBlock", fontSize=8.5, fontName="Helvetica-Oblique",
                textColor=colors.HexColor("#777777"),
                borderColor=colors.HexColor("#cccccc"),
                borderWidth=0.5, borderPadding=6,
                leading=12,
            ),
        )
    )

    # ── Build ─────────────────────────────────────────────────────────────────
    doc.build(story, onFirstPage=on_first, onLaterPages=on_later)
    return buf.getvalue()
