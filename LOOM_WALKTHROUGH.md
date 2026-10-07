# 2–3 Minute Loom Walkthrough Script
**Spotter Machine Learning Engineer Assessment: Freight Rate Prediction**

Use this timed script to record your 2–3 minute Loom video walkthrough. It addresses all five assessment prompt requirements in logical order.

---

## ⏱️ Video Outline & Timing Summary
- **[0:00 – 0:30] Introduction & Key Data Findings**
- **[0:30 – 1:05] Data Quality Issues & How They Were Resolved**
- **[1:05 – 1:40] Model Selection Reasoning & Target Formulation**
- **[1:40 – 2:15] Training & Validation Strategy (Data Split)**
- **[2:15 – 2:50] Code Walkthrough & Interactive UI Demo**
- **[2:50 – 3:00] Conclusion & December Chart Summary**

---

## 🎙️ Timed Script & Talking Points

### [0:00 – 0:30] Introduction & Key Data Findings
**Screen:** Show your terminal with `train.py` output or the Streamlit web application.

> "Hi Spotter team! Today I'm excited to walk you through my end-to-end machine learning solution for predicting truckload freight rates.
> 
> Starting with **Exploratory Data Analysis**, we examined 48,000 labeled historical loads across 72 freight markets and three equipment types—Dry Van, Reefer, and Flatbed.
> Three key findings stood out:
> 1. Freight rates scale primarily with distance, but short-haul loads under 350 miles carry substantially higher rates per mile ($2.50 to $3.50/mi) due to fixed loading and detention overheads.
> 2. Reefer and Flatbed equipment command consistent premiums over Dry Van.
> 3. Daily spot rates exhibit strong weekly cyclicality and pre-holiday capacity tightening."

---

### [0:30 – 1:05] Data-Quality Issues Identified & How We Addressed Them
**Screen:** Show the code snippet in `src/features.py` under `clean_dataframe`.

> "During data validation, I identified three important data quality issues and addressed each systematically:
> 
> 1. **Negative Weights:** We uncovered 292 loads in the training set and 145 in the validation set with negative cargo weights, like `-32,000` lbs. A distribution check confirmed these were sign-inversion entry errors rather than corrupted data. Taking `abs(weight)` restored the true weights without dropping data.
> 2. **Unseen Validation Markets:** Exactly 8 new cities appeared in `validation.csv` that were never in `train-test.csv` (including Chicago, Laredo, and Knoxville). A naive categorical model would fail on unseen categories. We resolved this by engineering latitude and longitude coordinates, great-circle Haversine distances, and directional transit bearings, allowing spatial interpolation across any new market.
> 3. **Missing Macro Market Index:** Missing values in `market_index` were resolved by constructing a daily calendar lookup across all 365 days, imputing missing values using each day's cross-load mean."

---

### [1:05 – 1:40] Reasoning Behind the Chosen Model & Target Formulation
**Screen:** Show the model architecture in `src/model.py` and the comparison table in `REPORT.md`.

> "For modeling, our key architectural decision was **reformulating the prediction target**:
> Rather than predicting total dollar rates directly—which suffers from severe heteroscedasticity across short vs. long hauls—we trained our models to predict **Rate-Per-Mile (RPM)**, then multiplied by haul distance. This single engineering choice reduced validation MAE from $178 down to $157.
> 
> For the model family, gradient boosted decision trees excel on tabular freight data. We built an **ensemble combining LightGBM (65% weight) and XGBoost (35% weight)**. LightGBM captures non-linear interactions between distance, circuity, and equipment type with extreme speed, while XGBoost adds regularization against outlier rates, yielding a **Holdout MAE of $157.09**, a **Median Absolute Error of $63.92**, and an **R² of 0.8186**."

---

### [1:40 – 2:15] Training and Validation Approach (How Data Was Split)
**Screen:** Show the validation results printed by `train.py`.

> "Our validation strategy mirrors real production conditions:
> Instead of using random K-Fold cross-validation—which creates future-to-past data leakage—we implemented a **Temporal Out-of-Time Holdout Split**:
> - We trained on **Months 1 through 8 (January to August, 38,477 loads)**.
> - We validated on **Months 9 and 10 (September and October, 9,523 loads)**.
> 
> Once validated, we retrained on the full development set to generate predictions for all 12,000 loads in `validation.csv` and the 31 fixed December loads, all of which pass Spotter's `score.py` validator."

---

### [2:15 – 2:50] Code Walkthrough & Interactive UI Demo
**Screen:** Open `http://localhost:8501` and demo the Streamlit application.

> "To make this solution operational for brokers and shippers, I created an interactive UI in `app.py`:
> - In the **Instant Rate Quoter**, you select any origin and destination—like Lexington to Fort Wayne.
> - It automatically populates the historical distance (360 miles), allows choosing equipment and date, and clicking 'Calculate' provides the estimated spot rate, rate-per-mile, and a **90% prediction confidence interval**.
> - There is also a **Batch CSV Predictor** for bulk quoting, and a **Market Insights tab** showing our December 2025 trajectory.
> - Our code is modularized cleanly under `src/features.py`, `src/model.py`, and `train.py`."

---

### [2:50 – 3:00] Conclusion & December Chart
**Screen:** Show `scorer_results/candidate_december.png`.

> "As you can see in the generated December chart, our model captures realistic mid-week surges, a pre-holiday tightening peak on Christmas Eve ($830.52), and year-end demand ($818.10), all within a solid economic envelope.
> Thank you for your time, and I look forward to your feedback!"
