# Spotter Freight Rate Prediction Challenge

Production-ready Machine Learning system and interactive quotation web interface developed for Spotter's Machine Learning Engineer assessment.

---

## 📌 Solution Highlights

- **Predictive Architecture**: High-performance Ensemble combining **LightGBM (65%)** and **XGBoost (35%)** trained on **Rate-Per-Mile (RPM)** rather than raw dollars, normalizing haul-length variance.
- **Holdout Validation Performance** (Months 9–10):
  - **MAE**: **$157.09**
  - **Median Absolute Error (MedAE)**: **$63.92**
  - **R² Score**: **0.8186**
  - **MAPE**: **7.42%**
- **Data Quality Remediation**:
  - Handled 292 negative weights in training and 145 in validation using `abs(weight)`.
  - Generalized to 8 unseen validation cities using geographic coordinates, Haversine distances, and directional transit bearings.
  - Imputed missing macro market conditions via 365-day calendar cross-load means.
- **Interactive Web Application**: Streamlit UI (`app.py`) for single-load instant quotation with 90% confidence intervals, batch CSV file processing, and December market insights.
- **Scorer Verification**: 100% compliant with `score.py`, validating 12,000 loads in `validation_predictions.csv` and 31 daily December loads in `data/december_chart_inputs.csv`.

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. Run Complete End-to-End Pipeline
Executes data preprocessing, temporal holdout validation, full model training, prediction generation, and runs `score.py` automatically:
```bash
python train.py
```

### 3. Run the Official Assessment Scorer
```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```
This validates all predictions and creates `scorer_results/candidate_december.png`.

### 4. Launch the Interactive Web Application (UI)
```bash
python -m streamlit run app.py
```
Open your browser to `http://localhost:8501`.

### 5. Generate Assessment PDF Report
```bash
python generate_report.py
```
Generates `Spotter_ML_Assessment_Report.pdf` with embedded charts and tables.

---

## 📂 Project Structure

```
Spotter_ML/
├── data/                                 # Standardized datasets directory
│   ├── train_test.csv                    # 48,000 development loads
│   ├── validation.csv                    # 12,000 validation loads
│   ├── december_chart_inputs.csv         # 31 December fixed input loads
│   └── validation_predictions.csv        # Completed final predictions
├── models/                               # Serialized model artifacts
│   └── freight_rate_model.pkl            # Production LightGBM + XGBoost ensemble
├── scorer_results/
│   └── candidate_december.png            # Official December chart produced by score.py
├── src/
│   ├── __init__.py
│   ├── features.py                       # Data cleaning, coordinates lookup, feature pipeline
│   └── model.py                          # FreightRateModel class with training & inference
├── app.py                                # Interactive Streamlit Web UI
├── generate_report.py                    # PDF report generator using ReportLab
├── requirements.txt                      # Project dependencies
├── score.py                              # Assessment validation script
├── train.py                              # End-to-end training & prediction CLI script
├── LOOM_WALKTHROUGH.md                   # 2-3 minute presentation script for Loom video
├── REPORT.md                             # Assessment technical report (markdown)
└── Spotter_ML_Assessment_Report.pdf      # Official assessment report (PDF)
```

---

## 📋 Deliverables & Submission Checklist

- [x] **Solution Code & Dependencies**: Complete Python codebase with clean modular design in `src/` and `train.py`.
- [x] **validation_predictions.csv**: Contains exactly 12,000 rows with columns `load_id,predicted_rate`.
- [x] **Candidate December Chart**: `scorer_results/candidate_december.png` validated via `score.py`.
- [x] **Assessment Report**: Available in both printable PDF (`Spotter_ML_Assessment_Report.pdf`) and GitHub Markdown (`REPORT.md`).
- [x] **Loom Video Script**: Step-by-step 2-3 minute recording script in `LOOM_WALKTHROUGH.md`.
- [x] **Interactive UI**: Fully functional web quoter in `app.py`.