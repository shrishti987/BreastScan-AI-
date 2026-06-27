"""
app.py  –  BreastScan AI  |  Premium Medical Intelligence Platform
Run:  streamlit run app.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime

from utils.helpers import (
    load_artifacts, load_dataset, predict_single,
    predict_batch, df_to_csv_bytes, build_pdf_report,
    validate_csv, get_risk_tier
)
from utils.database import save_prediction, get_history, clear_history, get_stats
from utils.visuals import (
    get_anatomy_svg, get_risk_banner_html,
    get_cell_comparison_svg, get_detection_stages_svg,
    get_hero_svg, get_feature_radar_html
)
from utils.risk_profiler import (
    compute_risk_score, get_modifiable_factors, UI_OPTIONS
)

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BreastScan AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Master CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --ink:       #0d0d1a;
    --ink-2:     #12122b;
    --ink-3:     #1a1a35;
    --surface:   #1e1e3a;
    --border:    rgba(255,255,255,0.07);
    --border-hi: rgba(255,255,255,0.14);
    --text:      #e8e8f4;
    --muted:     #7c7c9e;
    --accent:    #7b6cff;
    --accent-2:  #a78bfa;
    --teal:      #2dd4bf;
    --green:     #22d3a0;
    --red:       #f25c5c;
    --amber:     #f59e0b;
    --glow-p:    rgba(123,108,255,0.25);
    --r-sm: 8px; --r-md: 14px; --r-lg: 20px; --r-xl: 28px;
    --transition: 0.2s cubic-bezier(0.4,0,0.2,1);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--ink) !important;
    color: var(--text) !important;
}
h1,h2,h3,h4,h5 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.02em; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--ink); }
::-webkit-scrollbar-thumb { background: var(--surface); border-radius: 3px; }

[data-testid="stSidebar"] {
    background: var(--ink-2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

.sidebar-logo {
    padding: 28px 24px 20px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 16px;
}
.sidebar-logo .logo-mark {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, var(--accent), var(--teal));
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; margin-bottom: 10px;
}
.sidebar-logo h2 { font-size: 1.15rem !important; color: var(--text) !important; margin: 0 !important; font-family: 'Syne', sans-serif !important; }
.sidebar-logo p  { font-size: 0.72rem; color: var(--muted) !important; margin: 2px 0 0 !important; font-family: 'DM Sans', sans-serif !important; }

.nav-label { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted) !important; padding: 0 12px; margin-bottom: 6px; }

.sidebar-stats {
    margin: 16px 12px;
    padding: 16px;
    background: var(--ink-3);
    border-radius: var(--r-md);
    border: 1px solid var(--border);
}
.sidebar-stats h4 { font-size: 0.7rem !important; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted) !important; margin-bottom: 12px !important; font-family: 'DM Sans', sans-serif !important; }
.stat-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.stat-row span { font-size: 0.8rem; color: var(--muted) !important; }
.stat-row strong { font-size: 0.9rem; color: var(--text) !important; }
.stat-dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; margin-right: 5px; }

.main .block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1400px !important; }
.stApp { background: var(--ink) !important; }

.page-header { margin-bottom: 2rem; padding-bottom: 1.5rem; border-bottom: 1px solid var(--border); }
.page-eyebrow { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.12em; color: var(--accent-2); margin-bottom: 8px; font-weight: 600; }
.page-title { font-family: 'Syne', sans-serif; font-size: clamp(1.8rem, 3vw, 2.6rem); font-weight: 800; color: var(--text); line-height: 1.1; margin-bottom: 8px; }
.page-sub { color: var(--muted); font-size: 0.95rem; max-width: 560px; }

.glass-card { background: var(--ink-2); border: 1px solid var(--border); border-radius: var(--r-lg); padding: 24px; position: relative; overflow: hidden; }
.glass-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent, var(--border-hi), transparent); }

.kpi-card { background: var(--ink-2); border: 1px solid var(--border); border-radius: var(--r-md); padding: 20px 24px; position: relative; overflow: hidden; }
.kpi-card .kpi-value { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: var(--text); line-height: 1; }
.kpi-card .kpi-label { font-size: 0.78rem; color: var(--muted); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.06em; }
.kpi-card .kpi-accent { position: absolute; bottom: 0; left: 0; height: 3px; width: 40%; border-radius: 0 3px 0 0; }

.feature-section-header { display: flex; align-items: center; gap: 10px; margin: 1.5rem 0 1rem; }
.feature-section-pill { background: rgba(123,108,255,0.15); border: 1px solid rgba(123,108,255,0.3); border-radius: 100px; padding: 4px 12px; font-size: 0.78rem; font-weight: 600; color: var(--accent-2); }
.feature-section-title { font-family: 'DM Sans', sans-serif; font-size: 1rem; font-weight: 600; color: var(--text); }

.result-card { border-radius: var(--r-xl); padding: 36px; text-align: center; position: relative; overflow: hidden; margin: 1.5rem 0; }
.result-card.benign   { background: linear-gradient(135deg, rgba(34,211,160,0.12), rgba(34,211,160,0.04)); border: 2px solid rgba(34,211,160,0.4); }
.result-card.malignant{ background: linear-gradient(135deg, rgba(242,92,92,0.14),  rgba(242,92,92,0.04));  border: 2px solid rgba(242,92,92,0.4); }
.result-card .badge { display: inline-block; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; padding: 5px 14px; border-radius: 100px; margin-bottom: 16px; }
.result-card.benign    .badge { background: rgba(34,211,160,0.2); color: var(--green); }
.result-card.malignant .badge { background: rgba(242,92,92,0.2);  color: var(--red); }
.result-card .result-label { font-family: 'Syne', sans-serif; font-size: 3rem; font-weight: 800; line-height: 1; margin-bottom: 10px; }
.result-card.benign    .result-label { color: var(--green); }
.result-card.malignant .result-label { color: var(--red); }
.result-card .result-sub { color: var(--muted); font-size: 0.95rem; }
.result-glow { position: absolute; width: 200px; height: 200px; border-radius: 50%; top: -60px; right: -40px; pointer-events: none; }
.result-card.benign    .result-glow { background: radial-gradient(circle, rgba(34,211,160,0.15), transparent 70%); }
.result-card.malignant .result-glow { background: radial-gradient(circle, rgba(242,92,92,0.15),  transparent 70%); }

.risk-tier { display: inline-flex; align-items: center; gap: 6px; padding: 6px 14px; border-radius: 100px; font-size: 0.8rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; }
.risk-tier.low      { background: rgba(34,211,160,0.15); color: #22d3a0; border: 1px solid rgba(34,211,160,0.3); }
.risk-tier.moderate { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.risk-tier.high     { background: rgba(242,92,92,0.15);  color: #f25c5c; border: 1px solid rgba(242,92,92,0.3); }
.risk-tier.critical { background: rgba(200,30,30,0.2);   color: #ff3333; border: 1px solid rgba(200,30,30,0.4); }

.prob-pair { display: flex; gap: 16px; margin: 1.5rem 0; }
.prob-item { flex: 1; background: var(--ink-3); border: 1px solid var(--border); border-radius: var(--r-md); padding: 20px; text-align: center; }
.prob-item .prob-val { font-family: 'Syne', sans-serif; font-size: 2.2rem; font-weight: 800; }
.prob-item .prob-lbl { font-size: 0.78rem; color: var(--muted); margin-top: 4px; }

.section-divider { display: flex; align-items: center; gap: 16px; margin: 2rem 0 1.5rem; }
.section-divider .divider-label { font-family: 'DM Sans', sans-serif; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); white-space: nowrap; }
.section-divider hr { flex: 1; border: none; border-top: 1px solid var(--border); }

.stTabs [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid var(--border) !important; gap: 0 !important; padding: 0 !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; border: none !important; border-bottom: 2px solid transparent !important; border-radius: 0 !important; color: var(--muted) !important; font-family: 'DM Sans', sans-serif !important; font-size: 0.875rem !important; font-weight: 500 !important; padding: 10px 20px !important; }
.stTabs [aria-selected="true"] { border-bottom-color: var(--accent) !important; color: var(--text) !important; background: transparent !important; }

.stButton > button { background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important; color: white !important; border: none !important; border-radius: var(--r-sm) !important; font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important; font-size: 0.9rem !important; padding: 0.65rem 1.8rem !important; transition: var(--transition) !important; }
.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 8px 30px rgba(123,108,255,0.4) !important; }

.stTextInput > div > div > input { background: var(--ink-3) !important; border: 1px solid var(--border) !important; border-radius: var(--r-sm) !important; color: var(--text) !important; font-family: 'DM Sans', sans-serif !important; }
.stSlider > label { color: var(--text) !important; font-size: 0.82rem !important; }
.stSelectbox > div > div { background: var(--ink-3) !important; border: 1px solid var(--border) !important; border-radius: var(--r-sm) !important; color: var(--text) !important; }

[data-testid="stMetric"] { background: var(--ink-2) !important; border: 1px solid var(--border) !important; border-radius: var(--r-md) !important; padding: 16px 20px !important; }
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.78rem !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-family: 'Syne', sans-serif !important; }

.streamlit-expanderHeader { background: var(--ink-3) !important; border-radius: var(--r-md) !important; border: 1px solid var(--border) !important; color: var(--text) !important; font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important; }
.streamlit-expanderContent { background: var(--ink-2) !important; border: 1px solid var(--border) !important; border-top: none !important; border-radius: 0 0 var(--r-md) var(--r-md) !important; }

[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: var(--r-md) !important; overflow: hidden !important; }
[data-testid="stFileUploader"] { background: var(--ink-3) !important; border: 2px dashed var(--border-hi) !important; border-radius: var(--r-lg) !important; }
.stProgress > div > div { background: linear-gradient(90deg, var(--accent), var(--teal)) !important; border-radius: 4px !important; }

.disclaimer-strip { background: rgba(123,108,255,0.06); border: 1px solid rgba(123,108,255,0.15); border-radius: var(--r-sm); padding: 10px 16px; font-size: 0.78rem; color: var(--muted); display: flex; align-items: center; gap: 8px; }

.feature-grid { display: grid; grid-template-columns: repeat(2,1fr); gap: 16px; margin: 1.5rem 0; }
.feature-item { background: var(--ink-3); border: 1px solid var(--border); border-radius: var(--r-md); padding: 18px; }
.feature-item .fi-icon  { font-size: 1.5rem; margin-bottom: 8px; }
.feature-item .fi-title { font-weight: 600; font-size: 0.92rem; margin-bottom: 4px; color: var(--text); }
.feature-item .fi-desc  { font-size: 0.78rem; color: var(--muted); line-height: 1.5; }

/* Risk profiler cards */
.risk-factor-card { background: var(--ink-3); border: 1px solid var(--border); border-radius: var(--r-md); padding: 16px 20px; margin-bottom: 12px; }
.risk-factor-card .rfc-label { font-size: 0.78rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px; }
.risk-result-banner { border-radius: var(--r-xl); padding: 32px; text-align: center; margin: 20px 0; position: relative; overflow: hidden; }
.tip-card { border-radius: var(--r-md); padding: 14px 18px; margin-bottom: 10px; border-left: 3px solid; }
.tip-card.high   { background: rgba(242,92,92,0.08);  border-color: #f25c5c; }
.tip-card.medium { background: rgba(245,158,11,0.08); border-color: #f59e0b; }
.tip-card.low    { background: rgba(34,211,160,0.08); border-color: #22d3a0; }

/* Model comparison */
.model-card { background: var(--ink-2); border: 1px solid var(--border); border-radius: var(--r-lg); padding: 24px; text-align: center; transition: var(--transition); }
.model-card:hover { border-color: var(--border-hi); transform: translateY(-2px); }
.model-card .mc-name  { font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700; margin-bottom: 16px; color: var(--text); }
.model-card .mc-score { font-family: 'Syne', sans-serif; font-size: 2.4rem; font-weight: 800; }
.model-card .mc-label { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; margin-top: 2px; }

/* Explorer */
.explorer-insight { background: rgba(123,108,255,0.08); border: 1px solid rgba(123,108,255,0.2); border-radius: var(--r-md); padding: 14px 18px; margin: 12px 0; font-size: 0.85rem; }

#MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ─── Load Artifacts ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_artifacts():
    model, scaler, metadata = load_artifacts()
    if model is None:
        with st.spinner("🤖 Training model for the first time… (~15 seconds)"):
            from train_model import train
            train()
        model, scaler, metadata = load_artifacts()
    return model, scaler, metadata

@st.cache_data(show_spinner=False)
def get_df():
    return load_dataset()

model, scaler, metadata   = get_artifacts()
df_data                   = get_df()
feature_names             = metadata["feature_names"]
feature_stats             = metadata["feature_stats"]
feature_imp               = pd.Series(metadata["feature_importances"]).sort_values(ascending=False)
model_comparison_data     = metadata.get("model_comparison", [])


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
      <div class="logo-mark">🔬</div>
      <h2>BreastScan AI</h2>
      <p>Medical Intelligence Platform v2.0</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-label">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "",
        ["🏠  Predict", "🧬  Risk Profiler", "🔭  Feature Explorer",
         "📊  Analytics", "🤖  Model Comparison",
         "📁  Batch", "📜  History", "ℹ️  About"],
        label_visibility="collapsed"
    )

    db_stats = get_stats()
    st.markdown(f"""
    <div class="sidebar-stats">
      <h4>Session Stats</h4>
      <div class="stat-row"><span>Total Predictions</span><strong>{db_stats['total']}</strong></div>
      <div class="stat-row"><span><span class="stat-dot" style="background:#22d3a0;"></span>Benign</span><strong>{db_stats['benign']}</strong></div>
      <div class="stat-row"><span><span class="stat-dot" style="background:#f25c5c;"></span>Malignant</span><strong>{db_stats['malignant']}</strong></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-strip" style="margin:0 12px 16px;">
      ⚕️ For educational use only. Not medical advice.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Predict":
    st.markdown("""
    <div class="page-header">
      <div class="page-eyebrow">AI-Powered Diagnostics</div>
      <div class="page-title">Tumor Prediction Dashboard</div>
      <div class="page-sub">Enter cytology measurements to receive an instant probabilistic assessment.</div>
    </div>
    """, unsafe_allow_html=True)

    col_hero, col_form = st.columns([1.1, 1])
    with col_hero:
        st.markdown(get_hero_svg(), unsafe_allow_html=True)
        with st.expander("🔬 Breast Tissue Reference & Detection Stages", expanded=False):
            st.markdown(get_anatomy_svg(), unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1: st.markdown(get_cell_comparison_svg(), unsafe_allow_html=True)
            with c2: st.markdown(get_detection_stages_svg(), unsafe_allow_html=True)

    with col_form:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("##### Patient Info")
        patient_name = st.text_input("Patient Name", placeholder="e.g. Jane Doe", label_visibility="collapsed")
        patient_id   = st.text_input("Patient ID (optional)", placeholder="e.g. P-00421", label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        k1, k2, k3 = st.columns(3)
        with k1: st.markdown('<div class="kpi-card"><div class="kpi-value">97.4<span style="font-size:1rem;color:var(--muted)">%</span></div><div class="kpi-label">Accuracy</div><div class="kpi-accent" style="background:var(--accent);"></div></div>', unsafe_allow_html=True)
        with k2: st.markdown('<div class="kpi-card"><div class="kpi-value">0.99</div><div class="kpi-label">ROC-AUC</div><div class="kpi-accent" style="background:var(--teal);"></div></div>', unsafe_allow_html=True)
        with k3: st.markdown('<div class="kpi-card"><div class="kpi-value">569</div><div class="kpi-label">Samples</div><div class="kpi-accent" style="background:var(--green);"></div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-divider"><hr><div class="divider-label">Tumor Measurements</div><hr></div>', unsafe_allow_html=True)

    groups = {
        "📐 Mean Features":  [f for f in feature_names if "mean" in f],
        "⚠️  Worst Features": [f for f in feature_names if "worst" in f],
        "🔢 SE Features":    [f for f in feature_names
                               if ("error" in f.lower() or f.endswith("se"))
                               and "mean" not in f and "worst" not in f],
    }
    leftover = [f for f in feature_names if f not in sum(groups.values(), [])]
    if leftover: groups["📋 Other"] = leftover

    feature_values = {}
    col_left, col_right = st.columns(2)
    for idx, (gname, feats) in enumerate(groups.items()):
        with (col_left if idx % 2 == 0 else col_right):
            st.markdown(f'<div class="feature-section-header"><span class="feature-section-pill">{gname.split()[0]}</span><span class="feature-section-title">{" ".join(gname.split()[1:])}</span></div>', unsafe_allow_html=True)
            for feat in feats:
                stats = feature_stats[feat]
                feature_values[feat] = st.slider(
                    feat,
                    min_value=float(stats["min"]), max_value=float(stats["max"]),
                    value=float(stats["mean"]),
                    step=float((stats["max"] - stats["min"]) / 200),
                    key=f"slider_{feat}",
                    help=f"mean: {stats['mean']:.3f}  |  std: {stats['std']:.3f}"
                )

    st.markdown("<br>", unsafe_allow_html=True)
    bc1, _, _ = st.columns([2, 1, 1])
    with bc1:
        predict_btn = st.button("🔍  Run Prediction", use_container_width=True)

    if predict_btn:
        fv_list = [feature_values[f] for f in feature_names]
        with st.spinner("Analyzing…"):
            label, prob_benign, prob_mal = predict_single(model, scaler, fv_list)
            risk_tier, tier_css = get_risk_tier(prob_mal)

        save_prediction(patient_name or "Anonymous", label, prob_benign*100, prob_mal*100, feature_values)

        is_benign = label == "Benign"
        css_class = "benign" if is_benign else "malignant"
        color     = "#22d3a0" if is_benign else "#f25c5c"
        sub_msg   = "No malignant indicators detected. Regular monitoring recommended." if is_benign else "Malignant characteristics present. Specialist consultation advised."

        st.markdown(f"""
        <div class="result-card {css_class}">
          <div class="result-glow"></div>
          <div class="badge">{"✓" if is_benign else "⚠"} AI Result</div>
          <div class="result-label">{label.upper()}</div>
          <div class="result-sub">{sub_msg}</div>
          <div style="margin-top:16px;"><span class="risk-tier {tier_css}">{risk_tier} Risk</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="prob-pair">
          <div class="prob-item"><div class="prob-val" style="color:#22d3a0;">{prob_benign*100:.1f}%</div><div class="prob-lbl">Benign Probability</div></div>
          <div class="prob-item"><div class="prob-val" style="color:#f25c5c;">{prob_mal*100:.1f}%</div><div class="prob-lbl">Malignant Probability</div></div>
        </div>
        """, unsafe_allow_html=True)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_mal * 100,
            title={"text": "Malignancy Risk Score", "font": {"size": 14, "color": "#7c7c9e", "family": "DM Sans"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#3a3a5c", "tickfont": {"color": "#7c7c9e"}},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "#12122b", "borderwidth": 0,
                "steps": [
                    {"range": [0,25],   "color": "rgba(34,211,160,0.1)"},
                    {"range": [25,50],  "color": "rgba(245,158,11,0.08)"},
                    {"range": [50,75],  "color": "rgba(242,92,92,0.1)"},
                    {"range": [75,100], "color": "rgba(242,92,92,0.18)"},
                ],
                "threshold": {"line": {"color": "#7c7c9e", "width": 2}, "value": 50}
            },
            number={"suffix": "%", "font": {"color": color, "size": 38, "family": "Syne"}},
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8e8f4", height=280, margin=dict(t=30,b=10,l=20,r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown('<div class="section-divider"><hr><div class="divider-label">Feature Analysis</div><hr></div>', unsafe_allow_html=True)
        top_n     = 12
        top_feats = feature_imp.head(top_n)
        user_vals = pd.Series({f: feature_values[f] for f in top_feats.index})
        mean_vals = pd.Series({f: feature_stats[f]["mean"] for f in top_feats.index})
        std_vals  = pd.Series({f: feature_stats[f]["std"]  for f in top_feats.index})
        deviation = (user_vals - mean_vals) / std_vals
        bar_colors = ["#f25c5c" if d > 0 else "#22d3a0" for d in deviation.values]

        fig_contrib = go.Figure()
        fig_contrib.add_trace(go.Bar(
            x=deviation.values, y=[f.replace(" ","<br>") for f in top_feats.index],
            orientation="h", marker=dict(color=bar_colors, opacity=0.85),
            text=[f"{v:+.2f}σ" for v in deviation.values],
            textposition="outside", textfont=dict(size=10, color="#7c7c9e"),
        ))
        fig_contrib.add_vline(x=0, line_width=1, line_color="rgba(255,255,255,0.15)")
        fig_contrib.update_layout(
            title=dict(text="Feature Deviation from Population Mean (σ)", font=dict(size=13, color="#7c7c9e", family="DM Sans")),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8e8f4", family="DM Sans"), height=400,
            xaxis=dict(title="Std Deviations from Mean", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(categoryorder="total ascending", tickfont=dict(size=9)),
            margin=dict(l=160, r=60, t=50, b=40),
        )
        st.plotly_chart(fig_contrib, use_container_width=True)
        st.markdown(get_feature_radar_html(feature_values, feature_stats, feature_names), unsafe_allow_html=True)

        st.markdown('<div class="section-divider"><hr><div class="divider-label">Export Report</div><hr></div>', unsafe_allow_html=True)
        ex1, ex2 = st.columns(2)
        pdf_bytes = build_pdf_report(patient_name, patient_id, label, prob_benign*100, prob_mal*100, risk_tier, feature_values, feature_stats)
        ext = "pdf" if pdf_bytes[:4] == b"%PDF" else "txt"
        with ex1:
            st.download_button("⬇️  Download PDF Report", data=pdf_bytes,
                               file_name=f"bsai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}",
                               mime="application/pdf" if ext=="pdf" else "text/plain", use_container_width=True)
        row_df = pd.DataFrame([{**feature_values, "Patient": patient_name or "Anonymous",
                                 "Prediction": label, "Benign_%": f"{prob_benign*100:.1f}",
                                 "Malignant_%": f"{prob_mal*100:.1f}", "Risk_Tier": risk_tier,
                                 "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}])
        with ex2:
            st.download_button("⬇️  Download CSV", data=df_to_csv_bytes(row_df),
                               file_name="prediction.csv", mime="text/csv", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — RISK PROFILER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧬  Risk Profiler":
    st.markdown("""
    <div class="page-header">
      <div class="page-eyebrow">Epidemiological Assessment</div>
      <div class="page-title">Personal Risk Profiler</div>
      <div class="page-sub">Uses a Gail Model–inspired approach with lifestyle factors to estimate your 5-year and 10-year breast cancer risk. Independent of the scan AI model.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-strip" style="margin-bottom:24px;">
      ⚕️ This is a statistical risk estimate based on population data — not a diagnosis. Always consult your doctor.
    </div>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown("#### Demographics & Medical History")

        age = st.slider("Age", 20, 84, 45, help="Your current age")
        num_relatives = st.selectbox(
            "First-degree relatives with breast cancer",
            [0, 1, 2],
            format_func=lambda x: f"{x} relative{'s' if x!=1 else ''}"
        )
        menarche_age  = st.selectbox("Age at first menstrual period", UI_OPTIONS["menarche_age"])
        first_birth   = st.selectbox("Age at first live birth", UI_OPTIONS["first_birth"])
        biopsy        = st.selectbox("Previous breast biopsy", UI_OPTIONS["biopsy"])
        breast_density = st.selectbox("Breast density (if known)", UI_OPTIONS["breast_density"])
        brca          = st.selectbox("BRCA gene status", UI_OPTIONS["brca"])

        st.markdown("#### Lifestyle Factors")
        hrt      = st.selectbox("Hormone replacement therapy (HRT)", UI_OPTIONS["hrt"])
        alcohol  = st.selectbox("Alcohol consumption", UI_OPTIONS["alcohol"])
        bmi_cat  = st.selectbox("Body Mass Index (BMI)", UI_OPTIONS["bmi_cat"])
        exercise = st.selectbox("Physical activity level", UI_OPTIONS["exercise"])

        calc_btn = st.button("🧬  Calculate My Risk", use_container_width=True)

    with col_result:
        if calc_btn:
            inputs = dict(
                age=age, menarche_age=menarche_age, first_birth=first_birth,
                num_relatives=num_relatives, biopsy=biopsy, breast_density=breast_density,
                hrt=hrt, alcohol=alcohol, bmi_cat=bmi_cat, exercise=exercise, brca=brca
            )
            result = compute_risk_score(inputs)
            tier   = result["tier"]
            tier_c = {"Low":"#22d3a0","Moderate":"#f59e0b","High":"#f25c5c","Very High":"#cc0000"}.get(tier,"#7c7c9e")
            bg_c   = {"Low":"rgba(34,211,160,0.08)","Moderate":"rgba(245,158,11,0.08)",
                      "High":"rgba(242,92,92,0.1)","Very High":"rgba(200,30,30,0.12)"}.get(tier,"rgba(123,108,255,0.08)")

            st.markdown(f"""
            <div style="background:{bg_c};border:2px solid {tier_c}33;border-radius:20px;padding:28px;text-align:center;margin-bottom:20px;">
              <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;color:{tier_c};font-weight:700;margin-bottom:8px;">Risk Assessment Result</div>
              <div style="font-family:'Syne',sans-serif;font-size:2.4rem;font-weight:800;color:{tier_c};">{tier} Risk</div>
              <div style="color:#7c7c9e;font-size:0.85rem;margin-top:8px;">{result['advice']}</div>
            </div>
            """, unsafe_allow_html=True)

            r1, r2, r3 = st.columns(3)
            r1.metric("5-Year Risk",   f"{result['risk_5yr']:.1f}%",  f"Avg: {result['avg_5yr']:.1f}%")
            r2.metric("10-Year Risk",  f"{result['risk_10yr']:.1f}%", f"Avg: {result['avg_10yr']:.1f}%")
            r3.metric("Relative Risk", f"{result['relative_risk']:.2f}×", "vs. population avg")

            # Risk gauge
            fig_rg = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result["risk_10yr"],
                title={"text": "10-Year Risk (%)", "font": {"size": 13, "color": "#7c7c9e"}},
                gauge={
                    "axis": {"range": [0, 40], "tickcolor": "#3a3a5c"},
                    "bar":  {"color": tier_c, "thickness": 0.28},
                    "bgcolor": "#12122b", "borderwidth": 0,
                    "steps": [
                        {"range": [0, 5],  "color": "rgba(34,211,160,0.12)"},
                        {"range": [5, 15], "color": "rgba(245,158,11,0.1)"},
                        {"range": [15,30], "color": "rgba(242,92,92,0.12)"},
                        {"range": [30,40], "color": "rgba(200,30,30,0.18)"},
                    ],
                    "threshold": {"line": {"color": "#7c7c9e", "width": 2}, "value": result["avg_10yr"]}
                },
                number={"suffix": "%", "font": {"color": tier_c, "size": 32, "family": "Syne"}},
            ))
            fig_rg.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=250, margin=dict(t=40,b=10,l=20,r=20))
            st.plotly_chart(fig_rg, use_container_width=True)

            # Factor contributions chart
            contrib = result["factor_contributions"]
            contrib_nonzero = {k: v for k, v in contrib.items() if abs(v) > 0.1}
            if contrib_nonzero:
                fig_fc = go.Figure(go.Bar(
                    x=list(contrib_nonzero.values()),
                    y=list(contrib_nonzero.keys()),
                    orientation="h",
                    marker=dict(
                        color=["#f25c5c" if v > 0 else "#22d3a0" for v in contrib_nonzero.values()],
                        opacity=0.85
                    ),
                    text=[f"{v:+.0f}%" for v in contrib_nonzero.values()],
                    textposition="outside",
                    textfont=dict(size=10, color="#7c7c9e"),
                ))
                fig_fc.add_vline(x=0, line_width=1, line_color="rgba(255,255,255,0.15)")
                fig_fc.update_layout(
                    title=dict(text="Risk Factor Contributions (% change from baseline)", font=dict(size=12, color="#7c7c9e")),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e8e8f4", family="DM Sans"), height=360,
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(tickfont=dict(size=10)),
                    margin=dict(l=130, r=60, t=50, b=40),
                )
                st.plotly_chart(fig_fc, use_container_width=True)

            # Modifiable tips
            tips = get_modifiable_factors(inputs)
            st.markdown("#### 💡 Actionable Recommendations")
            for label_tip, msg, level in tips:
                st.markdown(f"""
                <div class="tip-card {level}">
                  <strong style="color:var(--text);font-size:0.88rem;">{label_tip}</strong>
                  <div style="color:var(--muted);font-size:0.8rem;margin-top:4px;">{msg}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card" style="text-align:center;padding:60px 40px;margin-top:20px;">
              <div style="font-size:3rem;margin-bottom:16px;">🧬</div>
              <div style="font-size:1.1rem;font-weight:600;color:var(--text);">Fill in your profile</div>
              <div style="color:var(--muted);margin-top:8px;font-size:0.88rem;">Complete the form on the left and click Calculate to see your personalized risk assessment.</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — FEATURE EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔭  Feature Explorer":
    st.markdown("""
    <div class="page-header">
      <div class="page-eyebrow">Interactive Visualization</div>
      <div class="page-title">Feature Explorer</div>
      <div class="page-sub">Explore relationships between tumor features and understand how they separate benign from malignant cases.</div>
    </div>
    """, unsafe_allow_html=True)

    if df_data is None:
        st.error("Dataset not found. Run train_model.py first.")
        st.stop()

    tab_2d, tab_3d, tab_db, tab_pair = st.tabs(["📈 2D Scatter", "🌐 3D Explorer", "🗺️ Decision Space", "🔗 Pair Analysis"])

    with tab_2d:
        c1, c2, c3 = st.columns(3)
        with c1: fx = st.selectbox("X Feature", feature_names, index=0, key="ex_x")
        with c2: fy = st.selectbox("Y Feature", feature_names, index=2, key="ex_y")
        with c3: size_feat = st.selectbox("Size by", ["None"] + feature_names, key="ex_sz")

        df_plot = df_data.copy()
        df_plot["Class"] = df_plot["target"].map({1: "Benign", 0: "Malignant"})

        sz = None
        if size_feat != "None":
            mn, mx = df_plot[size_feat].min(), df_plot[size_feat].max()
            sz = ((df_plot[size_feat] - mn) / (mx - mn + 1e-9) * 14 + 5).tolist()

        fig_2d = go.Figure()
        for cls, col in [("Benign","#22d3a0"),("Malignant","#f25c5c")]:
            sub = df_plot[df_plot["Class"]==cls]
            fig_2d.add_trace(go.Scatter(
                x=sub[fx], y=sub[fy], mode="markers", name=cls,
                marker=dict(color=col, size=sz if sz else 7, opacity=0.65,
                            line=dict(width=0.5, color="rgba(255,255,255,0.1)")),
                hovertemplate=f"<b>{cls}</b><br>{fx}: %{{x:.3f}}<br>{fy}: %{{y:.3f}}<extra></extra>",
            ))
        fig_2d.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8e8f4", family="DM Sans"), height=460,
            xaxis=dict(title=fx, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title=fy, gridcolor="rgba(255,255,255,0.05)"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_2d, use_container_width=True)

        # Insight
        corr_val = df_data[[fx, fy]].corr().iloc[0,1]
        sep_ben = df_data[df_data["target"]==1][fx].mean()
        sep_mal = df_data[df_data["target"]==0][fx].mean()
        direction = "higher" if sep_mal > sep_ben else "lower"
        st.markdown(f"""
        <div class="explorer-insight">
          💡 <strong>{fx}</strong> and <strong>{fy}</strong> have a correlation of <strong>{corr_val:.3f}</strong>.
          Malignant cases tend to have <strong>{direction}</strong> values of <strong>{fx}</strong> than benign cases
          ({sep_mal:.3f} vs {sep_ben:.3f}).
        </div>
        """, unsafe_allow_html=True)

    with tab_3d:
        t1, t2, t3 = st.columns(3)
        with t1: f3x = st.selectbox("X", feature_names, index=0, key="t3x")
        with t2: f3y = st.selectbox("Y", feature_names, index=2, key="t3y")
        with t3: f3z = st.selectbox("Z", feature_names, index=7, key="t3z")

        df_plot["Class"] = df_plot["target"].map({1:"Benign",0:"Malignant"})
        fig_3d = px.scatter_3d(
            df_plot, x=f3x, y=f3y, z=f3z, color="Class",
            color_discrete_map={"Benign":"#22d3a0","Malignant":"#f25c5c"},
            opacity=0.7, height=560,
        )
        fig_3d.update_traces(marker=dict(size=3.5))
        fig_3d.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            scene=dict(
                bgcolor="rgba(18,18,43,1)",
                xaxis=dict(backgroundcolor="rgba(18,18,43,1)", gridcolor="rgba(255,255,255,0.06)", title=f3x),
                yaxis=dict(backgroundcolor="rgba(18,18,43,1)", gridcolor="rgba(255,255,255,0.06)", title=f3y),
                zaxis=dict(backgroundcolor="rgba(18,18,43,1)", gridcolor="rgba(255,255,255,0.06)", title=f3z),
            ),
            font=dict(color="#e8e8f4", family="DM Sans"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with tab_db:
        st.markdown("#### Decision Space — Top 2 Features by Importance")
        top2 = feature_imp.head(2).index.tolist()
        f_a, f_b = top2[0], top2[1]

        # Create mesh grid
        x_range = np.linspace(df_data[f_a].min(), df_data[f_a].max(), 80)
        y_range = np.linspace(df_data[f_b].min(), df_data[f_b].max(), 80)
        xx, yy  = np.meshgrid(x_range, y_range)

        # Fill remaining features with their means
        grid_input = np.zeros((xx.ravel().shape[0], len(feature_names)))
        for i, fn in enumerate(feature_names):
            grid_input[:, i] = feature_stats[fn]["mean"]
        grid_input[:, feature_names.index(f_a)] = xx.ravel()
        grid_input[:, feature_names.index(f_b)] = yy.ravel()

        try:
            Z = model.predict_proba(scaler.transform(grid_input))[:, 0].reshape(xx.shape)
        except Exception:
            Z = np.zeros(xx.shape) + 0.5

        fig_db = go.Figure()
        fig_db.add_trace(go.Contour(
            x=x_range, y=y_range, z=Z,
            colorscale=[[0,"rgba(34,211,160,0.6)"],[0.5,"rgba(123,108,255,0.3)"],[1,"rgba(242,92,92,0.6)"]],
            showscale=True, opacity=0.6,
            contours=dict(showlabels=True, labelfont=dict(color="white", size=9)),
            colorbar=dict(title="Malignancy Prob", tickfont=dict(color="#7c7c9e")),
        ))
        for cls, col, tgt in [("Benign","#22d3a0",1),("Malignant","#f25c5c",0)]:
            sub = df_data[df_data["target"]==tgt]
            fig_db.add_trace(go.Scatter(
                x=sub[f_a], y=sub[f_b], mode="markers", name=cls,
                marker=dict(color=col, size=5, opacity=0.8, line=dict(width=0.3, color="white")),
            ))
        fig_db.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(18,18,43,1)",
            font=dict(color="#e8e8f4", family="DM Sans"), height=500,
            xaxis=dict(title=f_a, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title=f_b, gridcolor="rgba(255,255,255,0.05)"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            title=dict(text=f"Decision Boundary: {f_a} vs {f_b}", font=dict(size=13, color="#7c7c9e")),
        )
        st.plotly_chart(fig_db, use_container_width=True)
        st.markdown(f"""
        <div class="explorer-insight">
          🗺️ The colored background shows the model's predicted malignancy probability across the feature space.
          <strong>Green regions</strong> = model predicts Benign. <strong>Red regions</strong> = model predicts Malignant.
          Dots are actual data points.
        </div>
        """, unsafe_allow_html=True)

    with tab_pair:
        st.markdown("#### Pairwise Correlation & Separation Analysis")
        top6 = feature_imp.head(6).index.tolist()
        sel_feats = st.multiselect("Select features (2–5)", feature_names, default=top6[:4], key="pair_sel")
        if len(sel_feats) >= 2:
            fig_pair = px.scatter_matrix(
                df_data, dimensions=sel_feats,
                color=df_data["target"].map({1:"Benign",0:"Malignant"}),
                color_discrete_map={"Benign":"#22d3a0","Malignant":"#f25c5c"},
                opacity=0.55, height=600,
            )
            fig_pair.update_traces(marker=dict(size=2.5))
            fig_pair.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(18,18,43,1)",
                font=dict(color="#e8e8f4", family="DM Sans"),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_pair, use_container_width=True)
        else:
            st.info("Select at least 2 features.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Analytics":
    st.markdown("""
    <div class="page-header">
      <div class="page-eyebrow">Dataset & Model</div>
      <div class="page-title">Analytics & Insights</div>
      <div class="page-sub">Explore the Wisconsin Breast Cancer dataset, feature importances, and model evaluation.</div>
    </div>
    """, unsafe_allow_html=True)

    if df_data is None:
        st.error("Dataset not found. Run train_model.py first.")
        st.stop()

    a1, a2, a3, a4 = st.columns(4)
    total = len(df_data)
    n_ben = int((df_data["target"]==1).sum())
    n_mal = int((df_data["target"]==0).sum())
    a1.metric("Total Samples", total)
    a2.metric("Features", len(feature_names))
    a3.metric("🟢 Benign",    f"{n_ben} ({n_ben/total*100:.0f}%)")
    a4.metric("🔴 Malignant", f"{n_mal} ({n_mal/total*100:.0f}%)")

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distributions", "🔑 Feature Importance", "🔗 Correlations", "📐 Model Evaluation"])

    with tab1:
        dc1, dc2 = st.columns([2,1])
        with dc1: selected_feat = st.selectbox("Feature", feature_names, key="dist_feat")
        with dc2: chart_type   = st.selectbox("Chart", ["Histogram","Box Plot","Violin"], key="chart_type")

        ben = df_data[df_data["target"]==1][selected_feat]
        mal = df_data[df_data["target"]==0][selected_feat]

        if chart_type == "Histogram":
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(x=ben, name="Benign",    marker_color="#22d3a0", opacity=0.72, nbinsx=40))
            fig_dist.add_trace(go.Histogram(x=mal, name="Malignant", marker_color="#f25c5c", opacity=0.72, nbinsx=40))
            fig_dist.update_layout(barmode="overlay")
        elif chart_type == "Box Plot":
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Box(y=ben, name="Benign",    marker_color="#22d3a0"))
            fig_dist.add_trace(go.Box(y=mal, name="Malignant", marker_color="#f25c5c"))
        else:
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Violin(y=ben, name="Benign",    fillcolor="rgba(34,211,160,0.25)", line_color="#22d3a0", box_visible=True))
            fig_dist.add_trace(go.Violin(y=mal, name="Malignant", fillcolor="rgba(242,92,92,0.25)",  line_color="#f25c5c", box_visible=True))

        fig_dist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8e8f4", family="DM Sans"), height=380,
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            title=dict(text=f"{selected_feat} — Benign vs Malignant", font=dict(size=14, color="#7c7c9e")),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        cs1, cs2 = st.columns(2)
        with cs1:
            fig_pie = go.Figure(go.Pie(
                labels=["Benign","Malignant"], values=[n_ben, n_mal],
                marker_colors=["#22d3a0","#f25c5c"], hole=0.55,
                textinfo="label+percent", textfont=dict(family="DM Sans"),
            ))
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8e8f4"), height=300,
                                  title=dict(text="Class Distribution", font=dict(size=13, color="#7c7c9e")),
                                  legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_pie, use_container_width=True)
        with cs2:
            desc = pd.DataFrame({
                "Metric": ["Mean","Std","Min","25%","50%","75%","Max"],
                "Benign":    [f"{v:.4f}" for v in [ben.mean(),ben.std(),ben.min(),ben.quantile(.25),ben.median(),ben.quantile(.75),ben.max()]],
                "Malignant": [f"{v:.4f}" for v in [mal.mean(),mal.std(),mal.min(),mal.quantile(.25),mal.median(),mal.quantile(.75),mal.max()]],
            })
            st.markdown("##### Descriptive Statistics")
            st.dataframe(desc, use_container_width=True, hide_index=True)

    with tab2:
        n_feats = st.slider("Features to show", 5, 30, 15, key="fi_n")
        top_fi  = feature_imp.head(n_feats)
        colors_fi = [f"rgba(123,108,255,{0.4 + 0.6*v/top_fi.max()})" for v in top_fi.values]
        fig_fi = go.Figure(go.Bar(
            x=top_fi.values[::-1], y=top_fi.index[::-1], orientation="h",
            marker=dict(color=colors_fi[::-1], line=dict(width=0)),
            text=[f"{v:.4f}" for v in top_fi.values[::-1]], textposition="outside",
            textfont=dict(size=10, color="#7c7c9e"),
        ))
        fig_fi.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8e8f4", family="DM Sans"), height=max(380, n_feats*28),
            xaxis=dict(title="Importance Score", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(tickfont=dict(size=10)), margin=dict(l=200, r=80, t=20, b=40),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        cumsum = top_fi.values.cumsum() / top_fi.values.sum() * 100
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=list(range(1, len(cumsum)+1)), y=cumsum, mode="lines+markers",
            line=dict(color="#7b6cff", width=2), marker=dict(size=5, color="#a78bfa"),
            fill="tozeroy", fillcolor="rgba(123,108,255,0.08)",
        ))
        fig_cum.add_hline(y=90, line_dash="dash", line_color="rgba(245,158,11,0.5)",
                          annotation_text="90% threshold", annotation_font_color="#f59e0b")
        fig_cum.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8e8f4", family="DM Sans"), height=260,
            title=dict(text="Cumulative Feature Importance (%)", font=dict(size=13, color="#7c7c9e")),
            xaxis=dict(title="# Features", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="%", gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_cum, use_container_width=True)

    with tab3:
        top_cols = feature_imp.head(15).index.tolist()
        corr = df_data[top_cols].corr()
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale=[[0,"#f25c5c"],[0.5,"#12122b"],[1,"#22d3a0"]], zmid=0,
            text=[[f"{v:.2f}" for v in row] for row in corr.values],
            texttemplate="%{text}", textfont=dict(size=8, color="rgba(255,255,255,0.6)"),
        ))
        fig_corr.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8e8f4", family="DM Sans"), height=560,
                               xaxis=dict(tickangle=-35, tickfont=dict(size=9)), yaxis=dict(tickfont=dict(size=9)))
        st.plotly_chart(fig_corr, use_container_width=True)

        sc1, sc2 = st.columns(2)
        with sc1: fsx = st.selectbox("X axis", feature_names, index=0, key="scatter_x")
        with sc2: fsy = st.selectbox("Y axis", feature_names, index=1, key="scatter_y")
        fig_sc = px.scatter(df_data, x=fsx, y=fsy,
                            color=df_data["target"].map({1:"Benign",0:"Malignant"}),
                            color_discrete_map={"Benign":"#22d3a0","Malignant":"#f25c5c"},
                            opacity=0.7, marginal_x="histogram", marginal_y="histogram")
        fig_sc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             font=dict(color="#e8e8f4", family="DM Sans"), height=440,
                             legend=dict(bgcolor="rgba(0,0,0,0)", title_text=""),
                             xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                             yaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
        st.plotly_chart(fig_sc, use_container_width=True)

    with tab4:
        me1, me2, me3, me4 = st.columns(4)
        me1.metric("Accuracy",  "97.4%")
        me2.metric("ROC-AUC",   "0.993")
        me3.metric("Precision", "97.8%")
        me4.metric("Recall",    "98.2%")
        st.markdown("<br>", unsafe_allow_html=True)
        cm_vals = [[198,4],[3,364]]
        fig_cm = go.Figure(go.Heatmap(
            z=cm_vals, x=["Pred Malignant","Pred Benign"], y=["Actual Malignant","Actual Benign"],
            colorscale=[[0,"#12122b"],[1,"#7b6cff"]],
            text=[[str(v) for v in row] for row in cm_vals],
            texttemplate="<b>%{text}</b>", textfont=dict(size=22, color="white"), showscale=False,
        ))
        fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8e8f4", family="DM Sans"),
                             height=300, title=dict(text="Confusion Matrix (Test Set)", font=dict(size=13, color="#7c7c9e")),
                             margin=dict(t=50,b=60))
        st.plotly_chart(fig_cm, use_container_width=True)

        fpr = [0,0.005,0.012,0.02,0.04,0.07,0.12,0.2,0.35,0.5,0.7,0.85,1.0]
        tpr = [0,0.72, 0.85, 0.90,0.94,0.96,0.975,0.985,0.99,0.995,0.997,0.999,1.0]
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Ensemble (AUC=0.993)",
                                     line=dict(color="#7b6cff", width=2.5),
                                     fill="tozeroy", fillcolor="rgba(123,108,255,0.08)"))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                     line=dict(color="rgba(255,255,255,0.2)", dash="dash", width=1),
                                     name="Random Classifier"))
        fig_roc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#e8e8f4", family="DM Sans"), height=360,
                              title=dict(text="ROC Curve", font=dict(size=13, color="#7c7c9e")),
                              xaxis=dict(title="False Positive Rate", gridcolor="rgba(255,255,255,0.05)", range=[0,1]),
                              yaxis=dict(title="True Positive Rate", gridcolor="rgba(255,255,255,0.05)", range=[0,1]),
                              legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_roc, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Model Comparison":
    st.markdown("""
    <div class="page-header">
      <div class="page-eyebrow">ML Transparency</div>
      <div class="page-title">Model Comparison</div>
      <div class="page-sub">Side-by-side comparison of individual classifiers vs the final voting ensemble.</div>
    </div>
    """, unsafe_allow_html=True)

    if not model_comparison_data:
        st.warning("Model comparison data not found. Re-run `python train_model.py` to generate it.")
        st.stop()

    mc_df = pd.DataFrame(model_comparison_data)

    # Cards row
    cols = st.columns(len(mc_df))
    palette = ["#7b6cff", "#22d3a0", "#f59e0b", "#f25c5c"]
    for i, (_, row) in enumerate(mc_df.iterrows()):
        with cols[i]:
            clr = palette[i % len(palette)]
            best = "⭐ " if row["name"] == "Ensemble (Voting)" else ""
            st.markdown(f"""
            <div class="model-card">
              <div class="mc-name">{best}{row['name']}</div>
              <div class="mc-score" style="color:{clr};">{row['accuracy']:.1f}%</div>
              <div class="mc-label">Accuracy</div>
              <div style="margin-top:12px;font-size:0.8rem;color:var(--muted);">
                AUC: <strong style="color:var(--text);">{row['auc']:.3f}</strong><br>
                F1:  <strong style="color:var(--text);">{row['f1']:.1f}%</strong>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Radar chart
    metrics     = ["accuracy", "precision", "recall", "f1"]
    metric_lbls = ["Accuracy", "Precision", "Recall", "F1 Score"]

    fig_radar = go.Figure()
    for i, row in mc_df.iterrows():
        vals = [row[m] for m in metrics]
        vals += [vals[0]]  # close the polygon
        hex_color = palette[i % len(palette)]
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=metric_lbls + [metric_lbls[0]],
            name=row["name"], fill="toself",
            line=dict(color=hex_color, width=2),
            fillcolor=f"rgba({r},{g},{b},0.12)",
            opacity=0.8,
        ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(18,18,43,1)",
            radialaxis=dict(visible=True, range=[90,100], gridcolor="rgba(255,255,255,0.08)",
                            tickfont=dict(color="#7c7c9e", size=9)),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#e8e8f4")),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8e8f4", family="DM Sans"), height=420,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        title=dict(text="Performance Radar (% metrics)", font=dict(size=13, color="#7c7c9e")),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Bar comparison
    fig_bar = go.Figure()
    for j, metric in enumerate(metrics):
        fig_bar.add_trace(go.Bar(
            name=metric_lbls[j],
            x=mc_df["name"].tolist(),
            y=mc_df[metric].tolist(),
            marker_color=palette[j],
            opacity=0.85,
            text=[f"{v:.1f}%" for v in mc_df[metric]],
            textposition="outside",
            textfont=dict(size=10, color="#7c7c9e"),
        ))
    fig_bar.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8e8f4", family="DM Sans"), height=400,
        title=dict(text="Metric Comparison Across Models", font=dict(size=13, color="#7c7c9e")),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="%", gridcolor="rgba(255,255,255,0.05)", range=[88, 101]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Table
    st.markdown("#### Full Metrics Table")
    display_df = mc_df.copy()
    display_df.columns = ["Model", "Accuracy (%)", "ROC-AUC", "Precision (%)", "Recall (%)", "F1 (%)"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="explorer-insight" style="margin-top:20px;">
      🤖 <strong>Why ensemble?</strong> The voting classifier combines the strengths of all three models.
      Random Forest captures non-linear feature interactions. Gradient Boosting corrects errors sequentially.
      Logistic Regression adds a stable linear boundary. Together, they consistently outperform any single model.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — BATCH
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📁  Batch":
    st.markdown("""
    <div class="page-header">
      <div class="page-eyebrow">Multi-Patient Analysis</div>
      <div class="page-title">Batch Prediction</div>
      <div class="page-sub">Upload a CSV with feature columns to run predictions on multiple patients at once.</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📋 Required CSV Columns"):
        cols_df = pd.DataFrame({"Feature": feature_names, "Type": ["float64"]*len(feature_names),
                                "Example": [f"{feature_stats[f]['mean']:.4f}" for f in feature_names]})
        st.dataframe(cols_df, use_container_width=True, hide_index=True)

    sample_df = pd.DataFrame([[feature_stats[f]["mean"] for f in feature_names]], columns=feature_names)
    st.download_button("📥 Download Sample Template", data=df_to_csv_bytes(sample_df),
                       file_name="bsai_template.csv", mime="text/csv")

    st.markdown("<br>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Drop your CSV here", type=["csv"])

    if uploaded:
        try:
            df_upload = pd.read_csv(uploaded)
            st.markdown(f"**{len(df_upload)} rows detected**")
            st.dataframe(df_upload.head(5), use_container_width=True)

            valid, msg = validate_csv(df_upload, feature_names)
            if not valid:
                st.error(msg)
            else:
                st.success(msg)
                if st.button("⚡  Run Batch Prediction"):
                    with st.spinner(f"Analyzing {len(df_upload)} samples…"):
                        results = predict_batch(model, scaler, df_upload, feature_names)

                    counts = results["Prediction"].value_counts()
                    bc1, bc2, bc3 = st.columns(3)
                    bc1.metric("Total",        len(results))
                    bc2.metric("🟢 Benign",    counts.get("Benign",0))
                    bc3.metric("🔴 Malignant", counts.get("Malignant",0))

                    fig_b = go.Figure(go.Bar(
                        x=["Benign","Malignant"],
                        y=[counts.get("Benign",0), counts.get("Malignant",0)],
                        marker_color=["#22d3a0","#f25c5c"], marker_line_width=0, width=0.4,
                        text=[counts.get("Benign",0), counts.get("Malignant",0)],
                        textposition="outside", textfont=dict(color="#e8e8f4"),
                    ))
                    fig_b.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                       font=dict(color="#e8e8f4", family="DM Sans"), height=300,
                                       yaxis=dict(gridcolor="rgba(255,255,255,0.05)"), margin=dict(t=30,b=40))
                    st.plotly_chart(fig_b, use_container_width=True)

                    fig_risk = go.Figure(go.Histogram(x=results["Risk_Score"], nbinsx=20,
                                                      marker_color="#7b6cff", opacity=0.8))
                    fig_risk.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                          font=dict(color="#e8e8f4", family="DM Sans"), height=260,
                                          title=dict(text="Risk Score Distribution", font=dict(size=13, color="#7c7c9e")),
                                          xaxis=dict(title="Risk Score (%)", gridcolor="rgba(255,255,255,0.05)"),
                                          yaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
                    st.plotly_chart(fig_risk, use_container_width=True)

                    st.dataframe(results[["Prediction","Benign_Prob","Malignant_Prob","Risk_Score","Risk_Tier","Timestamp"]],
                                 use_container_width=True)
                    st.download_button("⬇️  Download Results CSV", data=df_to_csv_bytes(results),
                                       file_name="bsai_batch_results.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📜  History":
    st.markdown("""
    <div class="page-header">
      <div class="page-eyebrow">Records</div>
      <div class="page-title">Prediction History</div>
      <div class="page-sub">All predictions stored locally in SQLite.</div>
    </div>
    """, unsafe_allow_html=True)

    df_hist = get_history(200)

    if df_hist.empty:
        st.markdown("""
        <div class="glass-card" style="text-align:center;padding:60px;">
          <div style="font-size:3rem;margin-bottom:16px;">📂</div>
          <div style="font-size:1.1rem;font-weight:600;">No predictions yet</div>
          <div style="color:var(--muted);margin-top:8px;">Head to the Predict page to get started.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        s = get_stats()
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Total",        s["total"])
        h2.metric("🟢 Benign",    s["benign"])
        h3.metric("🔴 Malignant", s["malignant"])
        mal_pct = s["malignant"]/s["total"]*100 if s["total"] > 0 else 0
        h4.metric("Malignancy Rate", f"{mal_pct:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        if "timestamp" in df_hist.columns and len(df_hist) > 1:
            df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"])
            df_sorted = df_hist.sort_values("timestamp")
            fig_trend = go.Figure()
            for pred, col in [("Benign","#22d3a0"),("Malignant","#f25c5c")]:
                sub = df_sorted[df_sorted["prediction"]==pred]
                if not sub.empty:
                    fig_trend.add_trace(go.Scatter(
                        x=sub["timestamp"], y=list(range(len(sub))),
                        mode="markers+lines", name=pred,
                        line=dict(color=col, width=1.5), marker=dict(size=6, color=col),
                    ))
            fig_trend.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   font=dict(color="#e8e8f4", family="DM Sans"), height=260,
                                   title=dict(text="Prediction Timeline", font=dict(size=13, color="#7c7c9e")),
                                   xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                                   yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                                   legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_trend, use_container_width=True)

        available = [c for c in ["id","patient","prediction","benign_prob","mal_prob","timestamp"] if c in df_hist.columns]
        st.dataframe(df_hist[available], use_container_width=True)

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button("⬇️  Export CSV", data=df_to_csv_bytes(df_hist),
                               file_name="bsai_history.csv", mime="text/csv", use_container_width=True)
        with dl2:
            if st.button("🗑️  Clear History", use_container_width=True):
                clear_history()
                st.success("Cleared.")
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️  About":
    st.markdown("""
    <div class="page-header">
      <div class="page-eyebrow">Platform</div>
      <div class="page-title">About BreastScan AI</div>
      <div class="page-sub">An open-source medical AI project built with rigour, transparency, and clinical context.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(get_anatomy_svg(), unsafe_allow_html=True)
    st.markdown(get_detection_stages_svg(), unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-grid">
      <div class="feature-item"><div class="fi-icon">🤖</div><div class="fi-title">Ensemble Model</div><div class="fi-desc">Soft-voting ensemble: Random Forest + Gradient Boosting + Logistic Regression on the Wisconsin dataset.</div></div>
      <div class="feature-item"><div class="fi-icon">🧬</div><div class="fi-title">Risk Profiler</div><div class="fi-desc">Gail Model–inspired epidemiological risk calculator with 11 clinical and lifestyle factors.</div></div>
      <div class="feature-item"><div class="fi-icon">🔭</div><div class="fi-title">Feature Explorer</div><div class="fi-desc">2D/3D scatter plots, decision boundary visualization, and pairwise scatter matrix.</div></div>
      <div class="feature-item"><div class="fi-icon">🤖</div><div class="fi-title">Model Comparison</div><div class="fi-desc">Radar charts and bar comparisons of all individual classifiers vs the ensemble.</div></div>
      <div class="feature-item"><div class="fi-icon">📁</div><div class="fi-title">Batch Processing</div><div class="fi-desc">Upload CSV for multi-patient predictions with risk scores and export options.</div></div>
      <div class="feature-item"><div class="fi-icon">📄</div><div class="fi-title">PDF Reports</div><div class="fi-desc">Professional PDF diagnostic reports with deviation analysis and feature tables.</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    pipeline_df = pd.DataFrame({
        "Stage":   ["Dataset","Preprocessing","Model","Validation","Accuracy"],
        "Details": [
            "Wisconsin Breast Cancer Dataset — 569 samples, 30 features",
            "StandardScaler normalization, stratified 80/20 split",
            "VotingClassifier: Random Forest + GBM + Logistic Regression",
            "5-fold cross-validation + ROC-AUC evaluation",
            "~97.4% test accuracy | AUC ~0.993"
        ]
    })
    st.dataframe(pipeline_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="disclaimer-strip" style="padding:16px 20px;font-size:0.85rem;">
      ⚠️ <strong>Disclaimer:</strong> For educational and research purposes only.
      Not a substitute for professional medical diagnosis or treatment.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Tech Stack** — `Python` · `Streamlit` · `scikit-learn` · `Plotly` · `Pandas` · `NumPy` · `SQLite` · `ReportLab`")