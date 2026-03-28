import sys
from pathlib import Path

import cv2
import numpy as np

if __name__ == "__main__" and "streamlit" not in sys.modules:
    print("Run this app with Streamlit:")
    print("streamlit run frontend/streamlit.py")
    raise SystemExit(0)

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.adversarial_agent import adversarial_check
from backend.config import IMAGE_SIZE
from backend.decision_fusion_agent import fuse_decision
from backend.deep_learning_agent import deep_learning_detection
from backend.feature_agent import extract_features


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

/* ── Reset & Base ──────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #080c14;
    color: #c8d6e8;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2.5rem 2rem 4rem;
    max-width: 820px;
}

/* ── Scrollbar ─────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }

/* ── Hero Header ───────────────────────────────────────────── */
.hero {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-bottom: 2.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid #1a2a3f;
    position: relative;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -1px;
    left: 0;
    width: 80px;
    height: 2px;
    background: linear-gradient(90deg, #00d4ff, transparent);
}
.hero-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #00d4ff;
    opacity: 0.8;
}
.hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    line-height: 1.1;
    color: #eaf3ff;
    letter-spacing: -0.02em;
}
.hero-title span { color: #00d4ff; }
.hero-sub {
    font-size: 0.85rem;
    color: #5a7a9e;
    font-weight: 400;
    margin-top: 2px;
}

/* ── Upload Zone ───────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #0d1525;
    border: 1.5px dashed #1e3a5f;
    border-radius: 12px;
    padding: 0.5rem;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: #00d4ff44; }
[data-testid="stFileUploader"] label {
    color: #5a7a9e !important;
    font-size: 0.82rem !important;
}

/* ── Uploaded Image ────────────────────────────────────────── */
[data-testid="stImage"] img {
    border-radius: 10px;
    border: 1px solid #1a2a3f;
    box-shadow: 0 8px 32px rgba(0,212,255,0.06);
}

/* ── Button ────────────────────────────────────────────────── */
[data-testid="stButton"] > button {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    background: linear-gradient(135deg, #003d5c, #005f8a) !important;
    color: #00d4ff !important;
    border: 1px solid #00d4ff33 !important;
    border-radius: 8px !important;
    padding: 0.65rem 2rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 0 20px rgba(0,212,255,0.08) !important;
}
[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #005f8a, #0080b3) !important;
    border-color: #00d4ff88 !important;
    box-shadow: 0 0 28px rgba(0,212,255,0.18) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stButton"] > button:active { transform: translateY(0) !important; }

/* ── Section Label ─────────────────────────────────────────── */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #00d4ff;
    opacity: 0.7;
    margin-bottom: 0.9rem;
    margin-top: 1.6rem;
}

/* ── Score Cards ───────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #0d1525;
    border: 1px solid #1a2a3f;
    border-radius: 10px;
    padding: 1rem 1.2rem !important;
    position: relative;
    overflow: hidden;
}
[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #00d4ff, #0057ff);
    border-radius: 3px 0 0 3px;
}
[data-testid="stMetricLabel"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #3d6080 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.45rem !important;
    color: #7dd3f0 !important;
    font-weight: 700 !important;
}

/* ── Final Score card highlight ────────────────────────────── */
.final-score [data-testid="stMetric"] {
    border-color: #00d4ff33;
    background: #091825;
}
.final-score [data-testid="stMetric"]::before {
    background: linear-gradient(180deg, #00ffcc, #00d4ff);
}
.final-score [data-testid="stMetricValue"] {
    color: #00ffcc !important;
    font-size: 1.7rem !important;
}

/* ── Verdict Boxes ─────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.05em !important;
    border-left-width: 3px !important;
}

/* ── Adv Status text ───────────────────────────────────────── */
.adv-status {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #5a7a9e;
    background: #0d1525;
    border: 1px solid #1a2a3f;
    border-radius: 8px;
    padding: 0.65rem 1rem;
    margin: 0.6rem 0;
}
.adv-status strong { color: #a0c4e0; }

/* ── Expander ──────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #0d1525 !important;
    border: 1px solid #1a2a3f !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    color: #3d6080 !important;
    text-transform: uppercase !important;
}
[data-testid="stExpander"] p {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    color: #5a8aaa !important;
}

/* ── Info box ──────────────────────────────────────────────── */
[data-testid="stInfoBox"], [data-testid="stInfo"] {
    background: #0b1929 !important;
    border-color: #1a3a55 !important;
    border-radius: 10px !important;
    font-size: 0.82rem !important;
    color: #4a7090 !important;
}

/* ── Spinner ───────────────────────────────────────────────── */
[data-testid="stSpinner"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #00d4ff !important;
}

/* ── Divider ───────────────────────────────────────────────── */
hr {
    border-color: #1a2a3f !important;
    margin: 1.5rem 0 !important;
}
</style>
"""


def preprocess_uploaded_image(file_data):
    image_array = np.frombuffer(file_data, dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode uploaded image.")
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img / 255.0
    return img


def score_bar_html(score: float) -> str:
    """Render a thin horizontal bar representing a 0–1 score."""
    pct = int(score * 100)
    color = "#ff4d6d" if score > 0.6 else "#ffbe3d" if score > 0.35 else "#00d4ff"
    return f"""
    <div style="margin:4px 0 10px;font-family:'Space Mono',monospace;font-size:0.68rem;color:#3d6080;">
        <div style="background:#0d1525;border:1px solid #1a2a3f;border-radius:4px;height:6px;width:100%;overflow:hidden;">
            <div style="height:100%;width:{pct}%;background:linear-gradient(90deg,{color}99,{color});border-radius:4px;transition:width 0.6s;"></div>
        </div>
        <span style="color:#2a4a60;">{pct}%</span>
    </div>
    """


def run_app():
    st.set_page_config(
        page_title="Steganalysis · Detector",
        page_icon="🕵",
        layout="centered",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Hero ────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">Multi-Agent · Forensic System · v2.0</div>
        <div class="hero-title">Steganography<br><span>Detection</span></div>
        <div class="hero-sub">Upload an image to run the full agent pipeline — feature extraction,<br>deep learning inference, and adversarial validation.</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload ───────────────────────────────────────────────────
    st.markdown('<div class="section-label">① Upload Target Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Supported formats: PNG · JPG · JPEG",
        type=["png", "jpg", "jpeg"],
        label_visibility="visible",
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="", use_container_width=True)
        st.markdown('<div style="height:1.2rem;"></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-label">② Run Analysis</div>', unsafe_allow_html=True)
        run = st.button("⟶  Analyze Image", type="primary", use_container_width=True)

        if run:
            with st.spinner("Running agents — please wait..."):
                try:
                    image = preprocess_uploaded_image(uploaded_file.getvalue())

                    features    = extract_features(image)
                    feature_score = features["feature_score"]
                    dl_score    = deep_learning_detection(image)
                    adv         = adversarial_check(image)
                    adv_score   = adv["score"]
                    final_score, verdict = fuse_decision(feature_score, dl_score, adv_score)

                except Exception as exc:
                    st.error(f"Pipeline error: {exc}")
                    st.stop()

            # ── Results ──────────────────────────────────────────
            st.markdown('<div class="section-label">③ Agent Scores</div>', unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Feature Agent", f"{feature_score:.4f}")
                st.markdown(score_bar_html(feature_score), unsafe_allow_html=True)
            with col2:
                st.metric("Deep Learning", f"{dl_score:.4f}")
                st.markdown(score_bar_html(dl_score), unsafe_allow_html=True)
            with col3:
                st.metric("Adversarial", f"{adv_score:.4f}")
                st.markdown(score_bar_html(adv_score), unsafe_allow_html=True)

            # ── Final Score ───────────────────────────────────────
            st.markdown('<div class="section-label">④ Fusion Decision</div>', unsafe_allow_html=True)
            st.markdown('<div class="final-score">', unsafe_allow_html=True)
            st.metric("Final Fused Score", f"{final_score:.6f}")
            st.markdown(score_bar_html(final_score), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(
                f'<div class="adv-status"><strong>Adversarial Status:</strong> {adv["status"]}</div>',
                unsafe_allow_html=True,
            )

            # ── Verdict ───────────────────────────────────────────
            st.markdown('<div class="section-label">⑤ Verdict</div>', unsafe_allow_html=True)
            if verdict == "STEGO IMAGE DETECTED":
                st.error(f"🔴  {verdict}")
            else:
                st.success(f"🟢  {verdict}")

            # ── Extra Stats ───────────────────────────────────────
            with st.expander("▸  Pixel Statistics"):
                c1, c2 = st.columns(2)
                c1.metric("Mean", f"{features['mean']:.6f}")
                c2.metric("Variance", f"{features['variance']:.6f}")

    else:
        st.markdown(
            '<div style="font-family:\'Space Mono\',monospace;font-size:0.78rem;'
            'color:#2a4a60;padding:1rem 0;">→ Awaiting image upload to begin analysis.</div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    run_app()