"""
utils/helpers.py
Shared utility functions for the Streamlit app.
"""

import os
import io
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


# ─── Model Loading ──────────────────────────────────────────────────────────

def load_artifacts():
    """Load model, scaler, and metadata from disk."""
    model_path    = os.path.join(MODELS_DIR, "model.pkl")
    scaler_path   = os.path.join(MODELS_DIR, "scaler.pkl")
    meta_path     = os.path.join(MODELS_DIR, "metadata.pkl")

    if not all(os.path.exists(p) for p in [model_path, scaler_path, meta_path]):
        return None, None, None

    with open(model_path,  "rb") as f: model    = pickle.load(f)
    with open(scaler_path, "rb") as f: scaler   = pickle.load(f)
    with open(meta_path,   "rb") as f: metadata = pickle.load(f)
    return model, scaler, metadata


def load_dataset():
    """Load the saved training dataset."""
    path = os.path.join(MODELS_DIR, "dataset.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ─── Prediction ─────────────────────────────────────────────────────────────

def predict_single(model, scaler, feature_values: list):
    """
    Predict a single sample.
    Returns: (label_str, probability_benign, probability_malignant)
    """
    X = np.array(feature_values).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred  = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    label = "Benign" if pred == 1 else "Malignant"
    return label, float(proba[1]), float(proba[0])


def predict_batch(model, scaler, df: pd.DataFrame, feature_names: list):
    """
    Run predictions on a DataFrame with the required feature columns.
    Returns the DataFrame with added Prediction, Benign_Prob, Malignant_Prob columns.
    """
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[feature_names].values
    X_scaled = scaler.transform(X)
    preds  = model.predict(X_scaled)
    probas = model.predict_proba(X_scaled)

    result = df.copy()
    result["Prediction"]     = ["Benign" if p == 1 else "Malignant" for p in preds]
    result["Benign_Prob"]    = (probas[:, 1] * 100).round(2)
    result["Malignant_Prob"] = (probas[:, 0] * 100).round(2)
    result["Risk_Score"]     = (probas[:, 0] * 100).round(2)
    result["Timestamp"]      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return result


# ─── Export ─────────────────────────────────────────────────────────────────

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to CSV bytes for download."""
    return df.to_csv(index=False).encode("utf-8")


def build_pdf_report(patient_name: str, prediction: str, benign_prob: float,
                     malignant_prob: float, feature_values: dict) -> bytes:
    """
    Generate a simple PDF report using reportlab.
    Falls back to a text/plain byte string if reportlab is unavailable.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.units import cm

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                                topMargin=2*cm, bottomMargin=2*cm,
                                leftMargin=2.5*cm, rightMargin=2.5*cm)
        styles = getSampleStyleSheet()
        story  = []

        title_style = ParagraphStyle("title", parent=styles["Heading1"],
                                     fontSize=20, textColor=colors.HexColor("#1a1a2e"),
                                     spaceAfter=6)
        story.append(Paragraph("🔬 Breast Cancer Prediction Report", title_style))
        story.append(Spacer(1, 0.4*cm))

        meta = [
            ["Patient Name:", patient_name or "Anonymous"],
            ["Date & Time:", datetime.now().strftime("%B %d, %Y  %H:%M")],
            ["Prediction:", prediction],
            ["Benign Probability:",    f"{benign_prob:.1f}%"],
            ["Malignant Probability:", f"{malignant_prob:.1f}%"],
        ]
        t = Table(meta, colWidths=[5*cm, 10*cm])
        pred_color = colors.HexColor("#28a745") if prediction == "Benign" else colors.HexColor("#dc3545")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f4ff")),
            ("TEXTCOLOR",  (1, 2), (1, 2),  pred_color),
            ("FONTNAME",   (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE",   (0, 0), (-1, -1), 10),
            ("FONTNAME",   (0, 0), (0, -1),  "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
            ("TOPPADDING",  (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.8*cm))

        story.append(Paragraph("Input Feature Values", styles["Heading2"]))
        feat_data = [["Feature", "Value"]] + [[k, f"{v:.4f}"] for k, v in feature_values.items()]
        ft = Table(feat_data, colWidths=[9*cm, 6*cm])
        ft.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
            ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(ft)

        story.append(Spacer(1, 0.8*cm))
        disclaimer = (
            "<i>⚠️ Disclaimer: This report is generated by an AI model and is intended for "
            "informational purposes only. It is NOT a substitute for professional medical advice, "
            "diagnosis, or treatment. Always consult a qualified healthcare provider.</i>"
        )
        story.append(Paragraph(disclaimer, styles["Italic"]))
        doc.build(story)
        return buffer.getvalue()

    except ImportError:
        # Fallback plain-text report
        lines = [
            "BREAST CANCER PREDICTION REPORT",
            "=" * 40,
            f"Patient   : {patient_name or 'Anonymous'}",
            f"Date      : {datetime.now().strftime('%B %d, %Y  %H:%M')}",
            f"Prediction: {prediction}",
            f"Benign Prob   : {benign_prob:.1f}%",
            f"Malignant Prob: {malignant_prob:.1f}%",
            "",
            "Input Features:",
        ] + [f"  {k}: {v:.4f}" for k, v in feature_values.items()]
        return "\n".join(lines).encode("utf-8")


# ─── Validation ─────────────────────────────────────────────────────────────

def validate_csv(df: pd.DataFrame, feature_names: list):
    """
    Validate an uploaded CSV for batch prediction.
    Returns (is_valid: bool, message: str).
    """
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        return False, f"Missing {len(missing)} required column(s): {', '.join(missing[:5])}{'...' if len(missing)>5 else ''}"
    if df.empty:
        return False, "The uploaded file contains no data rows."
    non_numeric = [c for c in feature_names if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        return False, f"Non-numeric values found in: {', '.join(non_numeric[:3])}"
    return True, "✅ File looks good!"