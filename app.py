"""
app.py  –  Breast Cancer Prediction Web App
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

from utils.helpers import (
    load_artifacts, load_dataset, predict_single,
    predict_batch, df_to_csv_bytes, build_pdf_report, validate_csv
)
from utils.database import save_prediction, get_history, clear_history, get_stats
from utils.visuals import (
    get_anatomy_svg, get_risk_banner_svg,
    get_cell_comparison_svg, get_detection_stages_svg
)

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BreastScan AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

:root {
    --primary: #6c63ff;
    --benign: #22c55e;
    --malignant: #ef4444;
    --bg-card: rgba(255,255,255,0.05);
    --border: rgba(255,255,255,0.1);
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* Main background */
.main { background: #0a0a1a; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, rgba(108,99,255,0.15), rgba(108,99,255,0.05));
    border: 1px solid rgba(108,99,255,0.3);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    backdrop-filter: blur(10px);
}
.metric-card h2 { font-size: 2.2rem; margin: 0; }
.metric-card p  { color: #94a3b8; margin: 4px 0 0; font-size: 0.85rem; }

/* Result banner */
.result-benign {
    background: linear-gradient(135deg, rgba(34,197,94,0.2), rgba(34,197,94,0.05));
    border: 2px solid #22c55e;
    border-radius: 20px; padding: 28px; text-align: center;
}
.result-malignant {
    background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(239,68,68,0.05));
    border: 2px solid #ef4444;
    border-radius: 20px; padding: 28px; text-align: center;
}
.result-label { font-size: 2.5rem; font-weight: 700; font-family: 'Space Grotesk', sans-serif; }
.result-benign .result-label    { color: #22c55e; }
.result-malignant .result-label { color: #ef4444; }
.result-sub { color: #94a3b8; margin-top: 6px; font-size: 0.95rem; }

/* Section headers */
.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.3rem; font-weight: 600;
    color: #e2e8f0; margin: 1.5rem 0 0.8rem;
    border-left: 4px solid #6c63ff;
    padding-left: 12px;
}

/* Feature group card */
.feature-group {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 16px; margin-bottom: 12px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 12px; gap: 4px; padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; color: #94a3b8 !important;
    font-family: 'Inter', sans-serif; font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6c63ff, #a78bfa) !important;
    color: white !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6c63ff, #a78bfa);
    color: white; border: none; border-radius: 10px;
    font-weight: 600; padding: 0.6rem 1.5rem;
    transition: all 0.2s;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 8px 25px rgba(108,99,255,0.4); }

/* Slider labels */
.stSlider > label { color: #cbd5e1 !important; font-size: 0.88rem; }

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* Progress bar override */
.stProgress > div > div { background: linear-gradient(90deg, #6c63ff, #22c55e); }

/* Hide default streamlit elements */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Load Artifacts (with auto-train) ───────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_artifacts():
    model, scaler, metadata = load_artifacts()
    if model is None:
        with st.spinner("🤖 Training model for the first time... (this takes ~15 seconds)"):
            from train_model import train
            train()
        model, scaler, metadata = load_artifacts()
    return model, scaler, metadata


@st.cache_data(show_spinner=False)
def get_df():
    return load_dataset()


model, scaler, metadata = get_artifacts()
df_data = get_df()
feature_names   = metadata["feature_names"]
feature_stats   = metadata["feature_stats"]
feature_imp_raw = metadata["feature_importances"]
feature_imp = pd.Series(feature_imp_raw).sort_values(ascending=False)

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 BreastScan AI")
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("", ["🏠 Predict", "📊 Analytics", "📁 Batch Predict", "📜 History", "ℹ️ About"],
                    label_visibility="collapsed")
    st.markdown("---")
    db_stats = get_stats()
    st.markdown(f"**Total Predictions:** {db_stats['total']}")
    st.markdown(f"🟢 Benign: {db_stats['benign']}  |  🔴 Malignant: {db_stats['malignant']}")
    st.markdown("---")
    st.caption("⚕️ For educational use only. Not a substitute for medical advice.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT
# ═══════════════════════════════════════════════════════════════════════════
if page == "🏠 Predict":
    st.markdown("# 🔬 BreastScan AI — Prediction Dashboard")
    st.markdown("*Enter tumor measurement values below to receive an instant AI-powered prediction.*")

    patient_name = st.text_input("Patient Name (optional)", placeholder="e.g. Jane Doe", key="pname")

    # Anatomy illustration
    with st.expander("🔬 View Breast Anatomy Reference", expanded=False):
        st.markdown(get_anatomy_svg(), unsafe_allow_html=True)
        col_c, col_s = st.columns(2)
        with col_c:
            st.markdown(get_cell_comparison_svg(), unsafe_allow_html=True)
        with col_s:
            st.markdown(get_detection_stages_svg(), unsafe_allow_html=True)

    # ── Feature Groups ──────────────────────────────────────────────────
    groups = {
        "📐 Mean Features": [f for f in feature_names if "mean" in f],
        "⚠️  Worst Features": [f for f in feature_names if "worst" in f],
        "🔢 SE Features":   [f for f in feature_names if "error" in f.lower() or "se" in f.lower()
                              and "mean" not in f and "worst" not in f],
    }
    # catch-all for anything not grouped
    grouped_all = sum(groups.values(), [])
    leftover = [f for f in feature_names if f not in grouped_all]
    if leftover:
        groups["📋 Other Features"] = leftover

    feature_values = {}
    col_layout = st.columns(2)
    g_items = list(groups.items())

    for idx, (group_name, feats) in enumerate(g_items):
        with col_layout[idx % 2]:
            st.markdown(f'<div class="section-header">{group_name}</div>', unsafe_allow_html=True)
            for feat in feats:
                stats = feature_stats[feat]
                val = st.slider(
                    feat,
                    min_value=float(stats["min"]),
                    max_value=float(stats["max"]),
                    value=float(stats["mean"]),
                    step=float((stats["max"] - stats["min"]) / 200),
                    key=f"slider_{feat}",
                    help=f"Dataset mean: {stats['mean']:.3f} | std: {stats['std']:.3f}"
                )
                feature_values[feat] = val

    st.markdown("---")
    predict_col, _ = st.columns([1, 3])
    with predict_col:
        predict_btn = st.button("🔍 Run Prediction", use_container_width=True)

    if predict_btn:
        fv_list = [feature_values[f] for f in feature_names]
        label, prob_benign, prob_mal = predict_single(model, scaler, fv_list)

        # Save to DB
        save_prediction(patient_name, label, prob_benign * 100, prob_mal * 100, feature_values)

        st.markdown("---")
        # Visual result banner
        st.markdown(get_risk_banner_svg(label, prob_mal), unsafe_allow_html=True)

        r1, r2, r3 = st.columns([2, 1, 1])
        with r1:
            st.empty()
        with r2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Benign Prob.", f"{prob_benign*100:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)

        with r3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Malignant Prob.", f"{prob_mal*100:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Gauge chart ─────────────────────────────────────────────────
        st.markdown("#### Risk Probability Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob_mal * 100,
            delta={"reference": 50, "increasing": {"color": "#ef4444"}, "decreasing": {"color": "#22c55e"}},
            title={"text": "Malignancy Risk Score (%)", "font": {"size": 16, "color": "#e2e8f0"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                "bar": {"color": "#ef4444" if prob_mal > 0.5 else "#22c55e"},
                "bgcolor": "#1e1e3a",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30],  "color": "rgba(34,197,94,0.2)"},
                    {"range": [30, 60], "color": "rgba(234,179,8,0.2)"},
                    {"range": [60, 100], "color": "rgba(239,68,68,0.2)"},
                ],
                "threshold": {"line": {"color": "white", "width": 3}, "value": 50}
            },
            number={"suffix": "%", "font": {"color": "#e2e8f0", "size": 32}},
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", height=280
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── SHAP-style Feature Importance ───────────────────────────────
        st.markdown("#### 🔑 Top Feature Contributions")
        top_n = 10
        top_feats = feature_imp.head(top_n)
        user_vals  = pd.Series({f: feature_values[f] for f in top_feats.index})
        mean_vals  = pd.Series({f: feature_stats[f]["mean"] for f in top_feats.index})
        deviation  = ((user_vals - mean_vals) / pd.Series({f: feature_stats[f]["std"] for f in top_feats.index}))

        fig_contrib = go.Figure()
        colors_bar = ["#ef4444" if d > 0 else "#22c55e" for d in deviation.values]
        fig_contrib.add_trace(go.Bar(
            x=deviation.values,
            y=top_feats.index,
            orientation="h",
            marker_color=colors_bar,
            text=[f"{v:.2f}σ" for v in deviation.values],
            textposition="outside",
        ))
        fig_contrib.update_layout(
            title="Feature Deviation from Dataset Mean (std units)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e2e8f0"}, height=380,
            xaxis_title="Standard Deviations from Mean",
            yaxis={"categoryorder": "total ascending"},
            margin={"l": 180},
        )
        st.plotly_chart(fig_contrib, use_container_width=True)

        # ── Download report ─────────────────────────────────────────────
        st.markdown("#### 📄 Export Report")
        dc1, dc2 = st.columns(2)
        pdf_bytes = build_pdf_report(patient_name, label, prob_benign*100, prob_mal*100, feature_values)
        ext = "pdf" if pdf_bytes[:4] == b"%PDF" else "txt"
        with dc1:
            st.download_button("⬇️ Download PDF Report", data=pdf_bytes,
                               file_name=f"prediction_report.{ext}",
                               mime="application/pdf" if ext=="pdf" else "text/plain",
                               use_container_width=True)
        single_row = pd.DataFrame([{**feature_values, "Prediction": label,
                                     "Benign_Prob": f"{prob_benign*100:.1f}%",
                                     "Malignant_Prob": f"{prob_mal*100:.1f}%"}])
        with dc2:
            st.download_button("⬇️ Download CSV", data=df_to_csv_bytes(single_row),
                               file_name="prediction.csv", mime="text/csv",
                               use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics":
    st.markdown("# 📊 Dataset Analytics & Model Insights")

    if df_data is None:
        st.error("Dataset not found. Run `python train_model.py` first.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🔑 Feature Importance", "🔗 Correlations"])

    with tab1:
        st.markdown("### Feature Distributions by Class")
        selected_feat = st.selectbox("Select Feature", feature_names)
        ben  = df_data[df_data["target"] == 1][selected_feat]
        mal  = df_data[df_data["target"] == 0][selected_feat]
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=ben, name="Benign",    marker_color="#22c55e", opacity=0.7, nbinsx=40))
        fig_dist.add_trace(go.Histogram(x=mal, name="Malignant", marker_color="#ef4444", opacity=0.7, nbinsx=40))
        fig_dist.update_layout(
            barmode="overlay", title=f"Distribution: {selected_feat}",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e2e8f0"}, legend={"bgcolor": "rgba(0,0,0,0)"},
            xaxis_title=selected_feat, yaxis_title="Count"
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        # Class balance pie
        st.markdown("### Class Balance")
        counts = df_data["target"].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=["Benign", "Malignant"],
            values=[counts.get(1, 0), counts.get(0, 0)],
            marker_colors=["#22c55e", "#ef4444"],
            hole=0.45,
            textinfo="label+percent"
        ))
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e2e8f0"}, height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

    with tab2:
        st.markdown("### Top 15 Feature Importances (Random Forest)")
        top15 = feature_imp.head(15)
        fig_fi = go.Figure(go.Bar(
            x=top15.values[::-1], y=top15.index[::-1],
            orientation="h",
            marker=dict(
                color=top15.values[::-1],
                colorscale=[[0, "#302b63"], [1, "#6c63ff"]],
                showscale=False
            ),
            text=[f"{v:.4f}" for v in top15.values[::-1]],
            textposition="outside",
        ))
        fig_fi.update_layout(
            title="Feature Importance Scores",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e2e8f0"}, height=480,
            xaxis_title="Importance", margin={"l": 200}
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with tab3:
        st.markdown("### Correlation Matrix (Top 15 Features)")
        top_cols = feature_imp.head(15).index.tolist()
        corr = df_data[top_cols].corr()
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale="RdBu", zmid=0,
            text=[[f"{v:.2f}" for v in row] for row in corr.values],
            texttemplate="%{text}", textfont={"size": 8}
        ))
        fig_corr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e2e8f0"}, height=550
        )
        st.plotly_chart(fig_corr, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — BATCH PREDICT
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📁 Batch Predict":
    st.markdown("# 📁 Batch Prediction")
    st.markdown("Upload a CSV file containing feature columns for multiple patients.")

    with st.expander("📋 Required Columns"):
        st.write(feature_names)

    uploaded = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded:
        try:
            df_upload = pd.read_csv(uploaded)
            st.markdown(f"**Rows uploaded:** {len(df_upload)}")
            st.dataframe(df_upload.head(5), use_container_width=True)

            valid, msg = validate_csv(df_upload, feature_names)
            if not valid:
                st.error(msg)
            else:
                st.success(msg)
                if st.button("⚡ Run Batch Prediction", use_container_width=False):
                    with st.spinner("Predicting..."):
                        results = predict_batch(model, scaler, df_upload, feature_names)

                    st.markdown("### Results")
                    st.dataframe(
                        results[["Prediction", "Benign_Prob", "Malignant_Prob", "Risk_Score", "Timestamp"]],
                        use_container_width=True
                    )

                    counts = results["Prediction"].value_counts()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Samples",  len(results))
                    c2.metric("🟢 Benign",      counts.get("Benign", 0))
                    c3.metric("🔴 Malignant",   counts.get("Malignant", 0))

                    fig_b = go.Figure(go.Bar(
                        x=["Benign", "Malignant"],
                        y=[counts.get("Benign", 0), counts.get("Malignant", 0)],
                        marker_color=["#22c55e", "#ef4444"]
                    ))
                    fig_b.update_layout(
                        title="Prediction Distribution",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font={"color": "#e2e8f0"}, height=320
                    )
                    st.plotly_chart(fig_b, use_container_width=True)

                    st.download_button(
                        "⬇️ Download Results CSV",
                        data=df_to_csv_bytes(results),
                        file_name="batch_predictions.csv",
                        mime="text/csv",
                        use_container_width=False
                    )
        except Exception as e:
            st.error(f"Error processing file: {e}")

    else:
        # Download sample template
        sample_df = pd.DataFrame(
            [[feature_stats[f]["mean"] for f in feature_names]],
            columns=feature_names
        )
        st.download_button(
            "📥 Download Sample Template CSV",
            data=df_to_csv_bytes(sample_df),
            file_name="sample_template.csv",
            mime="text/csv"
        )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 — HISTORY
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📜 History":
    st.markdown("# 📜 Prediction History")

    df_hist = get_history(100)

    if df_hist.empty:
        st.info("No predictions recorded yet. Make a prediction on the **Predict** page.")
    else:
        s = get_stats()
        h1, h2, h3 = st.columns(3)
        h1.metric("Total", s["total"])
        h2.metric("🟢 Benign",    s["benign"])
        h3.metric("🔴 Malignant", s["malignant"])

        display_cols = ["id", "patient", "prediction", "benign_prob", "mal_prob", "timestamp"]
        available = [c for c in display_cols if c in df_hist.columns]
        st.dataframe(df_hist[available], use_container_width=True)

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button("⬇️ Export History CSV",
                               data=df_to_csv_bytes(df_hist),
                               file_name="prediction_history.csv",
                               mime="text/csv",
                               use_container_width=True)
        with dl2:
            if st.button("🗑️ Clear History", use_container_width=True):
                clear_history()
                st.success("History cleared.")
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown("# ℹ️ About BreastScan AI")
    st.markdown(get_anatomy_svg(), unsafe_allow_html=True)
    st.markdown(get_detection_stages_svg(), unsafe_allow_html=True)
    st.markdown("""
    ## 🔬 What is BreastScan AI?
    BreastScan AI is an interactive machine-learning web application for breast cancer prediction
    built with **Python**, **Streamlit**, and **scikit-learn**.

    ## 🤖 ML Pipeline
    | Stage | Details |
    |---|---|
    | Dataset | Wisconsin Breast Cancer Dataset (569 samples, 30 features) |
    | Preprocessing | StandardScaler normalization |
    | Model | Soft-voting Ensemble: Random Forest + GBM + Logistic Regression |
    | Typical Accuracy | ~97–98% on test set |
    | Evaluation | 5-fold cross-validation + ROC-AUC |

    ## 📋 Features
    - **Single prediction** with slider inputs for all 30 features
    - **Real-time risk gauge** and probability scores
    - **Feature deviation chart** (SHAP-inspired)
    - **Batch prediction** via CSV upload
    - **SQLite history** of all predictions
    - **Downloadable PDF & CSV reports**
    - **Dataset analytics** — distributions, feature importance, correlations

    ## ⚕️ Disclaimer
    > This tool is intended for **educational and research purposes only**.
    > It is **not** a substitute for professional medical diagnosis or treatment.
    > Always consult a licensed healthcare professional.

    ## 📦 Tech Stack
    `Python` · `Streamlit` · `scikit-learn` · `Plotly` · `Pandas` · `NumPy` · `SQLite` · `ReportLab`
    """)