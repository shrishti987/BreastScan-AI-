# 🔬 BreastScan AI — Breast Cancer Prediction Web App

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![ML](https://img.shields.io/badge/ML-Ensemble%20(RF%2BGBM%2BLR)-purple)

A full-featured, production-ready breast cancer prediction web application built with Python and Streamlit. Powered by an ensemble ML model achieving **~97–98% accuracy** on the Wisconsin Breast Cancer dataset.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎛️ Interactive Sliders | Input all 30 tumor features via intuitive sliders |
| 🟢🔴 Color-coded Results | Instant Benign/Malignant prediction with color feedback |
| 📊 Risk Gauge | Real-time probability gauge chart |
| 🔑 Feature Contributions | SHAP-inspired feature deviation chart |
| 📁 Batch Prediction | Upload CSV → bulk predictions → download results |
| 📄 PDF Reports | Exportable medical-style report per prediction |
| 📜 History | SQLite-backed prediction history with export |
| 📈 Analytics | Distributions, feature importance, correlation matrix |
| 🤖 Auto-Train | Model trains automatically on first launch |

---

## 🚀 Quick Start

### 1. Clone / Download

```bash
git clone https://github.com/your-username/breast-cancer-app.git
cd breast-cancer-app
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Pre-train the Model

```bash
python train_model.py
```

> The app will auto-train on first launch if models are missing.

### 5. Launch the App

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📁 Project Structure

```
breast_cancer_app/
├── app.py                  # Main Streamlit application
├── train_model.py          # Model training script
├── requirements.txt
├── README.md
├── utils/
│   ├── __init__.py
│   ├── helpers.py          # Prediction, export, validation utilities
│   └── database.py         # SQLite prediction history
├── models/                 # Auto-created on first run
│   ├── model.pkl           # Trained ensemble model
│   ├── scaler.pkl          # Feature scaler
│   ├── metadata.pkl        # Feature stats & importances
│   └── dataset.csv         # Training dataset
└── data/
    └── predictions.db      # SQLite history database
```

---

## 🤖 ML Model Details

- **Dataset**: [Wisconsin Breast Cancer Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
  - 569 samples, 30 numeric features, 2 classes (Benign / Malignant)
- **Preprocessing**: `StandardScaler` normalization
- **Model**: Soft-Voting Ensemble
  - Random Forest (200 trees) — weight 3
  - Gradient Boosting (150 estimators) — weight 2
  - Logistic Regression — weight 1
- **Evaluation**: 80/20 train-test split + 5-fold cross-validation
- **Typical Accuracy**: ~97–98% | **ROC-AUC**: ~0.99

---

## ☁️ Deployment

### Streamlit Cloud (Free)

1. Push your repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `app.py` as entry point → Deploy

### Render

1. Create a new **Web Service** on [render.com](https://render.com)
2. Build command: `pip install -r requirements.txt && python train_model.py`
3. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt && python train_model.py
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

---

## ⚕️ Disclaimer

> **This application is for educational and research purposes only.**  
> It is NOT a substitute for professional medical diagnosis, advice, or treatment.  
> Always consult a qualified healthcare professional for medical decisions.

---

## 📄 License

MIT License — feel free to use, modify, and distribute.