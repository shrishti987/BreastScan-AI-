"""
utils/helpers.py
Shared utility functions for BreastScan AI.
"""

import os
import io
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


# ─── Artifacts ───────────────────────────────────────────────────────────────

def load_artifacts():
    paths = [os.path.join(MODELS_DIR, f) for f in ("model.pkl", "scaler.pkl", "metadata.pkl")]
    if not all(os.path.exists(p) for p in paths):
        return None, None, None
    with open(paths[0], "rb") as f: model    = pickle.load(f)
    with open(paths[1], "rb") as f: scaler   = pickle.load(f)
    with open(paths[2], "rb") as f: metadata = pickle.load(f)
    return model, scaler, metadata


def load_dataset():
    path = os.path.join(MODELS_DIR, "dataset.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


# ─── Risk Tier ───────────────────────────────────────────────────────────────

def get_risk_tier(prob_mal: float):
    """
    Returns (tier_label, css_class) based on malignancy probability.
    """
    p = prob_mal * 100
    if p < 20:
        return "Low",      "low"
    elif p < 45:
        return "Moderate", "moderate"
    elif p < 70:
        return "High",     "high"
    else:
        return "Critical", "critical"


# ─── Prediction ──────────────────────────────────────────────────────────────

def predict_single(model, scaler, feature_values: list):
    X = np.array(feature_values).reshape(1, -1)
    X_s = scaler.transform(X)
    pred  = model.predict(X_s)[0]
    proba = model.predict_proba(X_s)[0]
    label = "Benign" if pred == 1 else "Malignant"
    return label, float(proba[1]), float(proba[0])


def predict_batch(model, scaler, df: pd.DataFrame, feature_names: list):
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    X = df[feature_names].values
    X_s   = scaler.transform(X)
    preds  = model.predict(X_s)
    probas = model.predict_proba(X_s)
    result = df.copy()
    result["Prediction"]     = ["Benign" if p == 1 else "Malignant" for p in preds]
    result["Benign_Prob"]    = (probas[:, 1] * 100).round(2)
    result["Malignant_Prob"] = (probas[:, 0] * 100).round(2)
    result["Risk_Score"]     = (probas[:, 0] * 100).round(2)
    result["Risk_Tier"]      = result["Risk_Score"].apply(lambda x: get_risk_tier(x/100)[0])
    result["Timestamp"]      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return result


# ─── Export ──────────────────────────────────────────────────────────────────

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def build_pdf_report(
    patient_name: str,
    patient_id: str,
    prediction: str,
    benign_prob: float,
    malignant_prob: float,
    risk_tier: str,
    feature_values: dict,
    feature_stats: dict,
) -> bytes:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table,
            TableStyle, HRFlowable
        )
        from reportlab.lib.units import cm
        from reportlab.lib.enums import TA_CENTER, TA_LEFT

        W, H = A4
        buf  = io.BytesIO()
        doc  = SimpleDocTemplate(
            buf, pagesize=A4,
            topMargin=2*cm, bottomMargin=2*cm,
            leftMargin=2.5*cm, rightMargin=2.5*cm
        )
        styles = getSampleStyleSheet()
        ink    = colors.HexColor("#0d0d1a")
        accent = colors.HexColor("#7b6cff")
        green  = colors.HexColor("#22d3a0")
        red    = colors.HexColor("#f25c5c")
        muted  = colors.HexColor("#7c7c9e")
        white  = colors.white
        light  = colors.HexColor("#f0f0fa")

        pred_color = green if prediction == "Benign" else red

        title_style = ParagraphStyle(
            "title", parent=styles["Heading1"],
            fontSize=22, textColor=ink, spaceAfter=4,
            fontName="Helvetica-Bold"
        )
        sub_style = ParagraphStyle(
            "sub", parent=styles["Normal"],
            fontSize=9, textColor=muted, spaceAfter=12
        )
        h2_style = ParagraphStyle(
            "h2", parent=styles["Heading2"],
            fontSize=13, textColor=ink, spaceBefore=14, spaceAfter=6,
            fontName="Helvetica-Bold"
        )

        story = []

        # Header
        story.append(Paragraph("BreastScan AI", title_style))
        story.append(Paragraph("Breast Cancer Prediction Report — Confidential", sub_style))
        story.append(HRFlowable(width="100%", thickness=1, color=accent, spaceAfter=12))

        # Patient + result summary
        tier_color_map = {"Low": green, "Moderate": colors.HexColor("#f59e0b"),
                          "High": red,  "Critical": colors.HexColor("#cc0000")}
        tier_c = tier_color_map.get(risk_tier, muted)

        meta = [
            ["Patient Name:", patient_name or "Anonymous",   "Report Date:", datetime.now().strftime("%B %d, %Y")],
            ["Patient ID:",   patient_id   or "—",           "Report Time:", datetime.now().strftime("%H:%M:%S")],
            ["",              "",                             "",             ""],
            ["Prediction:",   prediction,                    "Risk Tier:",   risk_tier],
            ["Benign Prob:",  f"{benign_prob:.1f}%",         "Malignant:",   f"{malignant_prob:.1f}%"],
        ]
        t = Table(meta, colWidths=[3.5*cm, 5.5*cm, 3.5*cm, 5.5*cm])
        ts = TableStyle([
            ("FONTNAME",  (0,0), (-1,-1), "Helvetica"),
            ("FONTSIZE",  (0,0), (-1,-1), 9),
            ("FONTNAME",  (0,0), (0,-1),  "Helvetica-Bold"),
            ("FONTNAME",  (2,0), (2,-1),  "Helvetica-Bold"),
            ("BACKGROUND",(0,0), (-1,-1), light),
            ("ROWBACKGROUNDS",(0,0),(-1,-1),[white, light]),
            ("GRID",      (0,0), (-1,-1), 0.4, colors.HexColor("#dddddd")),
            ("TOPPADDING",(0,0),(-1,-1), 6),
            ("BOTTOMPADDING",(0,0),(-1,-1), 6),
            ("TEXTCOLOR", (1,3),(1,3), pred_color),
            ("FONTNAME",  (1,3),(1,3), "Helvetica-Bold"),
            ("TEXTCOLOR", (3,3),(3,3), tier_c),
            ("FONTNAME",  (3,3),(3,3), "Helvetica-Bold"),
        ])
        t.setStyle(ts)
        story.append(t)
        story.append(Spacer(1, 0.6*cm))

        # Probability bar (text-based)
        story.append(Paragraph("Risk Probability Overview", h2_style))
        bar_w  = 13 * cm
        mal_w  = bar_w * (malignant_prob / 100)
        ben_w  = bar_w * (benign_prob   / 100)
        prob_table = [
            [Paragraph(f"Benign {benign_prob:.1f}%", ParagraphStyle("", fontSize=9, textColor=green, fontName="Helvetica-Bold")),
             Paragraph(f"Malignant {malignant_prob:.1f}%", ParagraphStyle("", fontSize=9, textColor=red, fontName="Helvetica-Bold", alignment=2))]
        ]
        pt = Table(prob_table, colWidths=[6.5*cm, 6.5*cm])
        pt.setStyle(TableStyle([("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2)]))
        story.append(pt)
        story.append(HRFlowable(width=f"{benign_prob}%", thickness=6, color=green, spaceAfter=2, lineCap="round"))
        story.append(Spacer(1, 0.4*cm))

        # Feature table
        story.append(Paragraph("Input Feature Values", h2_style))
        feat_header = [
            Paragraph("Feature", ParagraphStyle("", fontSize=8, textColor=white, fontName="Helvetica-Bold")),
            Paragraph("Value",   ParagraphStyle("", fontSize=8, textColor=white, fontName="Helvetica-Bold")),
            Paragraph("Mean",    ParagraphStyle("", fontSize=8, textColor=white, fontName="Helvetica-Bold")),
            Paragraph("Std Dev", ParagraphStyle("", fontSize=8, textColor=white, fontName="Helvetica-Bold")),
            Paragraph("Deviation", ParagraphStyle("", fontSize=8, textColor=white, fontName="Helvetica-Bold")),
        ]
        feat_data = [feat_header]
        for k, v in feature_values.items():
            stats  = feature_stats.get(k, {})
            mean   = stats.get("mean", 0)
            std    = stats.get("std",  1)
            dev    = (v - mean) / std if std else 0
            dev_s  = f"{dev:+.2f}σ"
            dev_c  = red if dev > 1 else (green if dev < -1 else muted)
            feat_data.append([
                k, f"{v:.4f}", f"{mean:.4f}", f"{std:.4f}",
                Paragraph(dev_s, ParagraphStyle("", fontSize=8, textColor=dev_c, fontName="Helvetica-Bold"))
            ])

        ft = Table(feat_data, colWidths=[5.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm])
        ft.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,0),  ink),
            ("FONTNAME",      (0,0),(-1,-1), "Helvetica"),
            ("FONTSIZE",      (0,0),(-1,-1), 8),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [white, light]),
            ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#dddddd")),
            ("TOPPADDING",    (0,0),(-1,-1), 4),
            ("BOTTOMPADDING", (0,0),(-1,-1), 4),
            ("ALIGN",         (1,0),(-1,-1), "CENTER"),
        ]))
        story.append(ft)
        story.append(Spacer(1, 0.8*cm))

        # Disclaimer
        story.append(HRFlowable(width="100%", thickness=0.5, color=muted, spaceAfter=10))
        disclaimer = ParagraphStyle("disc", parent=styles["Italic"], fontSize=8, textColor=muted, leading=12)
        story.append(Paragraph(
            "⚠ DISCLAIMER: This report is generated by an AI model (BreastScan AI) and is intended "
            "for educational and research purposes only. It is NOT a substitute for professional "
            "medical diagnosis, advice, or treatment. Always consult a qualified healthcare provider.",
            disclaimer
        ))

        doc.build(story)
        return buf.getvalue()

    except ImportError:
        lines = [
            "BSAI PREDICTION REPORT", "=" * 44,
            f"Patient   : {patient_name or 'Anonymous'}",
            f"Patient ID: {patient_id   or '—'}",
            f"Date      : {datetime.now().strftime('%B %d, %Y  %H:%M')}",
            f"Prediction: {prediction}",
            f"Risk Tier : {risk_tier}",
            f"Benign %  : {benign_prob:.1f}%",
            f"Malignant%: {malignant_prob:.1f}%",
            "", "Input Features:",
        ] + [f"  {k}: {v:.4f}" for k, v in feature_values.items()]
        return "\n".join(lines).encode("utf-8")


# ─── Validation ──────────────────────────────────────────────────────────────

def validate_csv(df: pd.DataFrame, feature_names: list):
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        return False, f"Missing {len(missing)} column(s): {', '.join(missing[:5])}{'…' if len(missing)>5 else ''}"
    if df.empty:
        return False, "Uploaded file has no data rows."
    non_num = [c for c in feature_names if not pd.api.types.is_numeric_dtype(df[c])]
    if non_num:
        return False, f"Non-numeric values in: {', '.join(non_num[:3])}"
    return True, f"✅ {len(df)} rows validated successfully."