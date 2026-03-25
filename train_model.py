"""
train_model.py
Trains an ensemble ML model on the breast cancer dataset,
saves the model, scaler, and feature metadata.
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
    accuracy_score, classification_report, roc_auc_score, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_data():
    """Load and return the breast cancer dataset as a DataFrame."""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target  # 1 = benign, 0 = malignant
    return df, data.feature_names.tolist(), data


def preprocess(df, feature_names):
    """Split and scale features."""
    X = df[feature_names].values
    y = df["target"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def build_ensemble():
    """Build a soft-voting ensemble classifier."""
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("lr", lr)],
        voting="soft",
        weights=[3, 2, 1]
    )
    return ensemble


def evaluate(model, X_test, y_test):
    """Print evaluation metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print(f"\n{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"{'='*50}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Malignant", "Benign"]))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    return acc, auc


def get_feature_importances(model, feature_names):
    """Extract feature importances from the RF sub-model."""
    rf_model = model.named_estimators_["rf"]
    importances = rf_model.feature_importances_
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    return fi


def save_artifacts(model, scaler, feature_names, feature_importances, raw_data):
    """Persist all artifacts needed by the Streamlit app."""
    with open(os.path.join(MODELS_DIR, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    df_raw = pd.DataFrame(raw_data.data, columns=feature_names)
    df_raw["target"] = raw_data.target
    df_raw.to_csv(os.path.join(MODELS_DIR, "dataset.csv"), index=False)

    metadata = {
        "feature_names": feature_names,
        "feature_importances": feature_importances.to_dict(),
        "target_names": raw_data.target_names.tolist(),
        "feature_stats": {
            feat: {
                "min": float(df_raw[feat].min()),
                "max": float(df_raw[feat].max()),
                "mean": float(df_raw[feat].mean()),
                "std": float(df_raw[feat].std()),
            }
            for feat in feature_names
        },
    }
    with open(os.path.join(MODELS_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    print(f"\n✅ Artifacts saved to: {MODELS_DIR}")


def train():
    print("🔬 Loading breast cancer dataset...")
    df, feature_names, raw_data = load_data()
    print(f"   Dataset shape: {df.shape}")

    print("⚙️  Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess(df, feature_names)

    print("🤖 Training ensemble model (RF + GBM + LR)...")
    model = build_ensemble()
    model.fit(X_train, y_train)

    print("📊 Evaluating model...")
    acc, auc = evaluate(model, X_test, y_test)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    print(f"\n5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    fi = get_feature_importances(model, feature_names)

    print("\n💾 Saving model artifacts...")
    save_artifacts(model, scaler, feature_names, fi, raw_data)
    return acc, auc


if __name__ == "__main__":
    train()