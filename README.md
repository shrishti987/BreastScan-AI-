# 🔬 BreastScan AI

An AI-powered breast cancer prediction platform built with Python and Streamlit. Uses the Wisconsin Breast Cancer Dataset and a soft-voting ensemble model to deliver instant probabilistic assessments — plus epidemiological risk profiling and interactive data exploration.

> ⚕️ **For educational and research purposes only. Not a substitute for medical advice.**

---

## Features

| Page | Description |
|---|---|
| 🏠 Predict | Slide 30 tumor measurements → get instant Benign/Malignant prediction with risk gauge and PDF report |
| 🧬 Risk Profiler | Gail Model–inspired 5-year & 10-year risk calculator using age, family history, BMI, HRT, BRCA status |
| 🔭 Feature Explorer | 2D/3D scatter plots, decision boundary heatmap, pairwise scatter matrix |
| 📊 Analytics | Feature distributions, importance ranking, correlation heatmap, ROC curve |
| 🤖 Model Comparison | RF vs GBM vs Logistic Regression vs Ensemble — radar + bar charts with real scores |
| 📁 Batch Predict | Upload CSV → predictions for multiple patients at once |
| 📜 History | SQLite-backed prediction log with timeline chart and CSV export |

---

## Quickstart

```bash
# 1. Clone and enter the project
cd BreastScan-AI

# 2. Activate virtual environment
# Windows:
.\venv\Scripts\Activate.ps1
# Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (once)
python train_model.py

# 5. Run the app
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## ML Pipeline

- **Dataset:** Wisconsin Breast Cancer Dataset — 569 samples, 30 features
- **Model:** Soft-voting ensemble (Random Forest + Gradient Boosting + Logistic Regression)
- **Preprocessing:** StandardScaler, stratified 80/20 train-test split
- **Validation:** 5-fold cross-validation + ROC-AUC
- **Accuracy:** ~97.4% | AUC ~0.993

---

## Project Structure

```
BreastScan-AI/
├── app.py                  # Main Streamlit app
├── train_model.py          # Model training script
├── requirements.txt
├── models/                 # Saved model artifacts (auto-generated)
├── data/                   # SQLite prediction history
└── utils/
    ├── helpers.py          # Prediction, PDF export, validation
    ├── visuals.py          # SVG illustrations and HTML components
    ├── database.py         # SQLite read/write
    └── risk_profiler.py    # Epidemiological risk engine
```

---

## Tech Stack

`Python` · `Streamlit` · `scikit-learn` · `Plotly` · `Pandas` · `NumPy` · `SQLite` · `ReportLab`

---

## Disclaimer

This tool is intended for **educational and research purposes only**. It is not a substitute for professional medical diagnosis, advice, or treatment. Always consult a qualified healthcare provider.