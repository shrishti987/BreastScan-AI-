"""
train_model.py
Trains an ensemble ML model on the Wisconsin Breast Cancer dataset.
Also saves individual model scores for the Model Comparison page.
Run standalone:  python train_model.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings("ignore")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_data():
    data = load_breast_cancer()
    df   = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df, data.feature_names.tolist(), data


def preprocess(df, feature_names):
    X = df[feature_names].values
    y = df["target"].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    return scaler.fit_transform(X_tr), scaler.transform(X_te), y_tr, y_te, scaler


def score_model(model, X_te, y_te, name="Model"):
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    return {
        "name":      name,
        "accuracy":  round(accuracy_score(y_te, y_pred) * 100, 2),
        "auc":       round(roc_auc_score(y_te, y_prob), 4),
        "precision": round(precision_score(y_te, y_pred) * 100, 2),
        "recall":    round(recall_score(y_te, y_pred) * 100, 2),
        "f1":        round(f1_score(y_te, y_pred) * 100, 2),
    }


def evaluate(model, X_te, y_te):
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_prob)
    print(f"\n{'='*50}\n  Accuracy : {acc:.4f}\n  ROC-AUC  : {auc:.4f}\n{'='*50}")
    print(classification_report(y_te, y_pred, target_names=["Malignant", "Benign"]))
    return acc, auc


def save_artifacts(model, scaler, feature_names, fi, raw_data, model_comparison):
    with open(os.path.join(MODELS_DIR, "model.pkl"),  "wb") as f: pickle.dump(model,  f)
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f: pickle.dump(scaler, f)

    df_raw = pd.DataFrame(raw_data.data, columns=feature_names)
    df_raw["target"] = raw_data.target
    df_raw.to_csv(os.path.join(MODELS_DIR, "dataset.csv"), index=False)

    metadata = {
        "feature_names":       feature_names,
        "feature_importances": fi.to_dict(),
        "target_names":        raw_data.target_names.tolist(),
        "model_comparison":    model_comparison,
        "feature_stats": {
            f: {
                "min":  float(df_raw[f].min()),
                "max":  float(df_raw[f].max()),
                "mean": float(df_raw[f].mean()),
                "std":  float(df_raw[f].std()),
            }
            for f in feature_names
        },
    }
    with open(os.path.join(MODELS_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)
    print(f"\n✅ Artifacts saved → {MODELS_DIR}")


def train():
    print("🔬 Loading dataset…")
    df, feature_names, raw_data = load_data()
    print(f"   Shape: {df.shape}")

    print("⚙️  Preprocessing…")
    X_tr, X_te, y_tr, y_te, scaler = preprocess(df, feature_names)

    print("🤖 Training individual models…")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

    rf.fit(X_tr, y_tr)
    gb.fit(X_tr, y_tr)
    lr.fit(X_tr, y_tr)

    model_comparison = [
        score_model(rf, X_te, y_te, "Random Forest"),
        score_model(gb, X_te, y_te, "Gradient Boosting"),
        score_model(lr, X_te, y_te, "Logistic Regression"),
    ]

    print("🤖 Training ensemble (RF + GBM + LR)…")
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("lr", lr)],
        voting="soft", weights=[3, 2, 1]
    )
    ensemble.fit(X_tr, y_tr)
    model_comparison.append(score_model(ensemble, X_te, y_te, "Ensemble (Voting)"))

    print("📊 Evaluating ensemble…")
    acc, auc = evaluate(ensemble, X_te, y_te)

    cv = cross_val_score(ensemble, X_tr, y_tr, cv=5, scoring="accuracy")
    print(f"\n5-Fold CV: {cv.mean():.4f} ± {cv.std():.4f}")

    fi = pd.Series(
        ensemble.named_estimators_["rf"].feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)

    print("\n💾 Saving artifacts…")
    save_artifacts(ensemble, scaler, feature_names, fi, raw_data, model_comparison)
    return acc, auc


if __name__ == "__main__":
    train()