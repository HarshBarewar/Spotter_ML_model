"""
Model architecture and training pipeline for Spotter Freight Rate Prediction.
Combines LightGBM and XGBoost in an ensemble trained on Rate-Per-Mile (RPM).
"""

from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
import lightgbm as lgb
import xgboost as xgb

from src.features import clean_dataframe, extract_features, KNOWN_CITIES, EQUIPMENT_BASE_RPM


class FreightRateModel:
    """
    Production-grade Freight Rate Prediction Model.
    Employs an ensemble of LightGBM and XGBoost trained on Rate-Per-Mile (RPM)
    with spatial, cyclical, and market interaction features.
    """

    def __init__(
        self,
        lgb_weight: float = 0.65,
        xgb_weight: float = 0.35,
        random_state: int = 42,
    ):
        self.lgb_weight = lgb_weight
        self.xgb_weight = xgb_weight
        self.random_state = random_state

        self.model_lgb: Optional[lgb.LGBMRegressor] = None
        self.model_xgb: Optional[xgb.XGBRegressor] = None

        self.daily_market_index_map: Dict[str, float] = {}
        self.lane_quote_signal_map: Dict[Tuple[str, str, str], float] = {}
        self.lane_typical_distances: Dict[Tuple[str, str], float] = {}
        self.category_maps: Dict[str, Dict[str, float]] = {}
        self.feature_columns: List[str] = []
        self.abs_error_percentiles: Tuple[float, float] = (15.0, 160.0)

    def _prepare_features(self, df: pd.DataFrame, is_training: bool = False) -> Tuple[pd.DataFrame, List[str]]:
        """Cleans and extracts features, encoding categorical attributes."""
        cleaned = clean_dataframe(
            df,
            daily_market_index_map=self.daily_market_index_map,
            lane_quote_signal_map=self.lane_quote_signal_map,
        )
        engineered = extract_features(cleaned)

        # Base numerical features
        base_features = [
            'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon',
            'haversine_dist', 'circuity', 'delta_lat', 'delta_lon', 'mid_lat', 'mid_lon', 'bearing',
            'distance', 'weight', 'weight_per_mile', 'is_heavy', 'is_short_haul', 'is_long_haul',
            'month', 'day', 'dayofweek', 'dayofyear', 'is_weekend', 'is_month_end',
            'sin_doy', 'cos_doy', 'sin_dow', 'cos_dow',
            'quote_signal', 'market_index', 'expected_cost', 'expected_cost_market',
            'market_adjusted_quote', 'signal_spread',
        ]

        # Target/Frequency encoding for categoricals
        cat_cols = ['pickup', 'delivery', 'equipment']
        for col in cat_cols:
            freq_col = f'{col}_freq'
            if is_training:
                freq = engineered[col].value_counts(normalize=True).to_dict()
                self.category_maps[col] = freq
            freq_map = self.category_maps.get(col, {})
            engineered[freq_col] = engineered[col].map(freq_map).fillna(0.001).astype(float)
            base_features.append(freq_col)

        return engineered, base_features

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Fits the ensemble on training data.
        Builds lookup maps and optimizes gradient boosting regressors.
        """
        # Build reference lookup tables
        t_df = train_df.copy()
        t_df['date'] = pd.to_datetime(t_df['date'])

        # Daily market index lookup from train data
        train_daily_mi = t_df.groupby(t_df['date'].dt.strftime('%Y-%m-%d'))['market_index'].mean().dropna().to_dict()
        self.daily_market_index_map.update(train_daily_mi)

        # Lane quote signals and distances
        for _, row in t_df.iterrows():
            k_lane_eq = (row['pickup'], row['delivery'], row['equipment'])
            k_lane = (row['pickup'], row['delivery'])
            if k_lane_eq not in self.lane_quote_signal_map and pd.notna(row['quote_signal']):
                self.lane_quote_signal_map[k_lane_eq] = float(row['quote_signal'])
            if k_lane not in self.lane_typical_distances and pd.notna(row['distance']):
                self.lane_typical_distances[k_lane] = float(row['distance'])

        train_eng, feature_cols = self._prepare_features(train_df, is_training=True)
        self.feature_columns = feature_cols

        X_train = train_eng[feature_cols].copy()
        y_train_rpm = (train_eng['posted_rate'] / train_eng['distance']).copy()

        # 1. Fit LightGBM
        self.model_lgb = lgb.LGBMRegressor(
            n_estimators=900,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        self.model_lgb.fit(X_train, y_train_rpm)

        # 2. Fit XGBoost
        self.model_xgb = xgb.XGBRegressor(
            n_estimators=900,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model_xgb.fit(X_train, y_train_rpm)

        metrics = {}
        if val_df is not None:
            metrics = self.evaluate(val_df)
            pred_res = self.predict(val_df)
            val_errors = np.abs(val_df['posted_rate'].to_numpy() - pred_res['predicted_rate'].to_numpy())
            self.abs_error_percentiles = (float(np.percentile(val_errors, 10)), float(np.percentile(val_errors, 90)))

        return metrics

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs inference on any DataFrame of loads.
        Returns predicted_rate, predicted_rpm, lower_bound, and upper_bound.
        """
        if self.model_lgb is None or self.model_xgb is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")

        test_eng, _ = self._prepare_features(df, is_training=False)
        X_test = test_eng[self.feature_columns]

        # Predict RPM
        pred_lgb_rpm = self.model_lgb.predict(X_test)
        pred_xgb_rpm = self.model_xgb.predict(X_test)

        pred_rpm = (self.lgb_weight * pred_lgb_rpm) + (self.xgb_weight * pred_xgb_rpm)
        # Ensure positive RPM
        pred_rpm = np.clip(pred_rpm, 0.50, 15.00)

        # Scale by distance
        distances = test_eng['distance'].to_numpy(dtype=float)
        predicted_rates = pred_rpm * distances

        # Confidence bounds (90% prediction interval)
        margin = self.abs_error_percentiles[1]
        lower_bound = np.clip(predicted_rates - margin, 50.0, None)
        upper_bound = predicted_rates + margin

        results = pd.DataFrame({
            'predicted_rate': np.round(predicted_rates, 2),
            'predicted_rpm': np.round(pred_rpm, 3),
            'lower_bound': np.round(lower_bound, 2),
            'upper_bound': np.round(upper_bound, 2),
        }, index=df.index)

        if 'load_id' in df.columns:
            results.insert(0, 'load_id', df['load_id'])

        return results

    def evaluate(self, val_df: pd.DataFrame) -> Dict[str, float]:
        """Calculates regression metrics on labeled validation data."""
        preds = self.predict(val_df)
        y_true = val_df['posted_rate'].to_numpy(dtype=float)
        y_pred = preds['predicted_rate'].to_numpy(dtype=float)

        mae = float(mean_absolute_error(y_true, y_pred))
        medae = float(median_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)

        return {
            'mae': mae,
            'medae': medae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
        }

    def save(self, filepath: str) -> None:
        """Serializes the model to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> FreightRateModel:
        """Loads a serialized model from disk."""
        return joblib.load(filepath)
