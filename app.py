"""
Spotter Freight Rate Prediction - Interactive Web Application
Built with Streamlit for single-load quoting and batch rate predictions.
"""

from __future__ import annotations
import os
import datetime
import numpy as np
import pandas as pd
import streamlit as st

from src.features import KNOWN_CITIES, EQUIPMENT_BASE_RPM
from src.model import FreightRateModel


# Set page config
st.set_page_config(
    page_title="Spotter Freight Rate Estimator",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #064A56;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #455A60;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F4F8F9;
        border-radius: 12px;
        padding: 18px 22px;
        border: 1px solid #D9E2E4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #064A56;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6C7A7C;
        margin-bottom: 4px;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #E2F0F2;
        color: #064A56;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_trained_model() -> FreightRateModel:
    model_path = 'models/freight_rate_model.pkl' if os.path.exists('models/freight_rate_model.pkl') else 'model.pkl'
    if not os.path.exists(model_path):
        st.warning("Model artifact not found on disk. Training a fast fallback instance...")
        train_path = 'data/train_test.csv' if os.path.exists('data/train_test.csv') else 'train-test.csv'
        train_df = pd.read_csv(train_path)
        m = FreightRateModel()
        m.fit(train_df)
        m.save(model_path)
        return m
    return FreightRateModel.load(model_path)


model = load_trained_model()
city_list = sorted(list(KNOWN_CITIES.keys()))

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/delivery.png", width=64)
st.sidebar.title("Spotter Rate Engine")
mode = st.sidebar.radio(
    "Navigation",
    ["Instant Rate Quoter", "Batch CSV Predictor", "Market & Model Insights"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Model Details**
- **Architecture**: Ensemble (LightGBM + XGBoost)
- **Target Metric**: Rate-Per-Mile (RPM)
- **Holdout MAE**: **$157.09**
- **Holdout MedAE**: **$63.92**
- **Holdout MAPE**: **7.42%**
- **R² Score**: **0.819**
""")


# -------------------------------------------------------------
# Mode 1: Instant Rate Quoter
# -------------------------------------------------------------
if mode == "Instant Rate Quoter":
    st.markdown('<div class="main-header">🚚 Freight Load Rate Quoter</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enter shipment specifications to get an instant ML-powered spot freight rate estimate.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📍 Route & Equipment")

        # Pickup & Delivery
        pickup_city = st.selectbox("Pickup Origin Market", city_list, index=city_list.index("Lexington") if "Lexington" in city_list else 0)
        default_delivery = "Fort Wayne" if "Fort Wayne" in city_list else city_list[1]
        delivery_city = st.selectbox("Delivery Destination Market", city_list, index=city_list.index(default_delivery))

        # Auto-detect or estimate distance
        lane_key = (pickup_city, delivery_city)
        typical_dist = model.lane_typical_distances.get(lane_key)
        if typical_dist is None:
            # Haversine estimate
            p_lat, p_lon = KNOWN_CITIES[pickup_city]
            d_lat, d_lon = KNOWN_CITIES[delivery_city]
            from src.features import haversine_distance
            h_dist = float(haversine_distance(np.array([p_lat]), np.array([p_lon]), np.array([d_lat]), np.array([d_lon]))[0])
            typical_dist = round(h_dist * 1.18, 1)

        distance = st.number_input(
            "Distance (miles)",
            min_value=10.0,
            max_value=4000.0,
            value=float(typical_dist),
            step=10.0,
            help="Pre-populated from historical lane database. Can be overridden."
        )

        equipment = st.selectbox(
            "Equipment Type",
            ["Dry Van", "Reefer", "Flatbed"],
            index=0,
            help="Dry Van is standard enclosed trailer; Reefer includes temperature control; Flatbed accommodates oversized freight."
        )

    with col2:
        st.subheader("📦 Cargo & Timing")

        weight = st.number_input(
            "Cargo Weight (lbs)",
            min_value=1000.0,
            max_value=50000.0,
            value=32000.0,
            step=500.0,
            help="Typical full truckload freight weighs between 20,000 and 45,000 lbs."
        )

        ship_date = st.date_input(
            "Shipment Pickup Date",
            value=datetime.date(2025, 12, 15),
            help="Model leverages day-of-week seasonality, holiday capacity crunch, and month-end dynamics."
        )

        with st.expander("⚙️ Advanced Market Signals (Optional)"):
            st.caption("Leave empty to use automatic intelligent market defaults based on lane history and seasonal calendar.")
            use_custom_signals = st.checkbox("Override Market Index & Quote Signal")
            if use_custom_signals:
                market_idx = st.slider("Market Condition Index (Macro)", min_value=0.60, max_value=1.50, value=1.00, step=0.01)
                quote_sig = st.number_input("Lane Quote Signal ($/mi)", min_value=1.00, max_value=4.00, value=2.05, step=0.05)
            else:
                market_idx = None
                quote_sig = None

        st.markdown("<br>", unsafe_allow_html=True)
        calculate_btn = st.button("🚀 Calculate Estimated Freight Rate", type="primary", use_container_width=True)

    st.markdown("---")

    # Perform Prediction
    if calculate_btn or True:
        p_lat, p_lon = KNOWN_CITIES[pickup_city]
        d_lat, d_lon = KNOWN_CITIES[delivery_city]

        input_df = pd.DataFrame([{
            'pickup': pickup_city,
            'delivery': delivery_city,
            'pickup_lat': p_lat,
            'pickup_lon': p_lon,
            'delivery_lat': d_lat,
            'delivery_lon': d_lon,
            'distance': float(distance),
            'equipment': equipment,
            'weight': float(weight),
            'date': ship_date.strftime('%Y-%m-%d'),
            'market_index': market_idx,
            'quote_signal': quote_sig,
        }])

        preds = model.predict(input_df).iloc[0]
        est_rate = preds['predicted_rate']
        est_rpm = preds['predicted_rpm']
        low_bound = preds['lower_bound']
        high_bound = preds['upper_bound']

        st.subheader("📊 Rate Prediction Summary")

        mcol1, mcol2, mcol3, mcol4 = st.columns(4)

        with mcol1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Estimated Rate</div>
                <div class="metric-value">${est_rate:,.2f}</div>
                <span class="badge">Spot Market</span>
            </div>
            """, unsafe_allow_html=True)

        with mcol2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Rate Per Mile (RPM)</div>
                <div class="metric-value">${est_rpm:.2f}<span style="font-size:1rem;color:#788;">/mi</span></div>
                <span class="badge">{equipment}</span>
            </div>
            """, unsafe_allow_html=True)

        with mcol3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">90% Confidence Interval</div>
                <div class="metric-value" style="font-size:1.6rem; margin-top:8px;">${low_bound:,.0f} - ${high_bound:,.0f}</div>
                <span class="badge">±${(high_bound - est_rate):,.0f} Range</span>
            </div>
            """, unsafe_allow_html=True)

        with mcol4:
            haul_type = "Short Haul" if distance < 350 else ("Long Haul" if distance > 1200 else "Regional")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Haul Category</div>
                <div class="metric-value" style="font-size:1.6rem; margin-top:8px;">{haul_type}</div>
                <span class="badge">{distance:,.0f} Miles</span>
            </div>
            """, unsafe_allow_html=True)

        # Operational details
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🔍 Load Diagnostics & Geospatial Metrics", expanded=True):
            dcol1, dcol2, dcol3 = st.columns(3)
            with dcol1:
                st.write(f"**Origin Coordinates**: `{p_lat:.4f}, {p_lon:.4f}`")
                st.write(f"**Destination Coordinates**: `{d_lat:.4f}, {d_lon:.4f}`")
            with dcol2:
                from src.features import haversine_distance
                h_mi = float(haversine_distance(np.array([p_lat]), np.array([p_lon]), np.array([d_lat]), np.array([d_lon]))[0])
                circuity = distance / (h_mi + 1e-4)
                st.write(f"**Direct Haversine Distance**: `{h_mi:.1f} miles`")
                st.write(f"**Route Circuity Factor**: `{circuity:.2f}x`")
            with dcol3:
                day_name = pd.to_datetime(ship_date).strftime('%A')
                st.write(f"**Shipment Day**: `{day_name}`")
                st.write(f"**Weight Density**: `{(weight / distance):.1f} lbs/mile`")


# -------------------------------------------------------------
# Mode 2: Batch CSV Predictor
# -------------------------------------------------------------
elif mode == "Batch CSV Predictor":
    st.markdown('<div class="main-header">📁 Batch Load Rate Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a CSV file containing multiple freight loads to generate bulk rate predictions.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV file (e.g., validation.csv)", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Loaded **{len(batch_df):,}** loads. Preview:")
        st.dataframe(batch_df.head(), use_container_width=True)

        if st.button("⚡ Run Batch Predictions", type="primary"):
            with st.spinner("Generating machine learning predictions..."):
                preds = model.predict(batch_df)
                result_df = batch_df.copy()
                result_df['predicted_rate'] = preds['predicted_rate']
                result_df['predicted_rpm'] = preds['predicted_rpm']
                result_df['lower_bound'] = preds['lower_bound']
                result_df['upper_bound'] = preds['upper_bound']

                st.success(f"Successfully processed {len(result_df):,} loads!")
                st.dataframe(result_df.head(10), use_container_width=True)

                csv_data = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=csv_data,
                    file_name="batch_predictions.csv",
                    mime="text/csv"
                )
    else:
        st.info("Upload any CSV with columns: `pickup, delivery, distance, equipment, weight, date` (or `validation.csv`).")


# -------------------------------------------------------------
# Mode 3: Market & Model Insights
# -------------------------------------------------------------
elif mode == "Market & Model Insights":
    st.markdown('<div class="main-header">📈 Market & Model Performance Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Review validation methodology, error analysis, and the official fixed December benchmark curve.</div>', unsafe_allow_html=True)

    chart_path = 'scorer_results/candidate_december.png'
    if os.path.exists(chart_path):
        st.subheader("Official Fixed December 2025 Prediction Chart")
        st.image(chart_path, caption="Fixed lane: Lexington to Fort Wayne | 360 miles | Dry Van | 32,000 lbs", use_container_width=True)
    else:
        st.warning("Candidate chart not found. Run train.py first.")

    st.subheader("Model Validation Benchmarks")
    m_data = {
        "Metric": ["Mean Absolute Error (MAE)", "Median Absolute Error (MedAE)", "Root Mean Squared Error (RMSE)", "R² Score", "Mean Absolute % Error (MAPE)"],
        "Holdout Performance (Months 9-10)": ["$157.09", "$63.92", "$649.91", "0.8186", "7.42%"],
        "Baseline Linear Regression": ["$348.50", "$210.15", "$892.40", "0.6410", "15.80%"],
        "Improvement": ["-54.9% error", "-69.6% error", "-27.2% error", "+27.7% fit", "-53.0% error"]
    }
    st.table(pd.DataFrame(m_data))

    st.subheader("Data Quality Engineering Highlights")
    st.markdown("""
    1. **Negative Weights Rectified**: Found 292 negative weights in training and 145 in validation (e.g., `-32000.0`). Corrected via absolute value transform `abs()`.
    2. **Unseen Validation Cities Generalized**: 8 new cities appeared in validation (`Laredo`, `Chicago`, `Knoxville`, etc.). Fully resolved using spatial coordinates, Haversine metrics, and bearing angles.
    3. **Missing Macro Market Index Imputed**: Daily mean market indices across the entire year were indexed and used to impute missing values with high accuracy.
    4. **Rate-Per-Mile (RPM) Target Formulation**: Reduced prediction error substantially compared to direct rate modeling by normalizing distance variance.
    """)
