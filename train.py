"""
End-to-end training and evaluation script for Spotter Freight Rate Prediction.
"""

from __future__ import annotations
import os
import subprocess
import sys
import numpy as np
import pandas as pd

from src.features import KNOWN_CITIES
from src.model import FreightRateModel


def main():
    print("=" * 70)
    print("SPOTTER FREIGHT RATE PREDICTION - END-TO-END PIPELINE")
    print("=" * 70)

    # 1. Load Data
    train_path = 'data/train_test.csv' if os.path.exists('data/train_test.csv') else 'train-test.csv'
    val_path = 'data/validation.csv' if os.path.exists('data/validation.csv') else 'validation.csv'
    dec_path = 'data/december_chart_inputs.csv' if os.path.exists('data/december_chart_inputs.csv') else 'december-chart-inputs.csv'

    print(f"\n[1/6] Loading datasets...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    dec_df = pd.read_csv(dec_path)

    print(f"  - Training loads:   {len(train_df):,} rows (Dates: {train_df['date'].min()} to {train_df['date'].max()})")
    print(f"  - Validation loads: {len(val_df):,} rows (Dates: {val_df['date'].min()} to {val_df['date'].max()})")
    print(f"  - December loads:   {len(dec_df):,} rows (Dates: {dec_df['date'].min()} to {dec_df['date'].max()})")

    # 2. Extract macro daily market indices across full calendar
    all_dates = pd.concat([train_df[['date', 'market_index']], val_df[['date', 'market_index']]])
    all_dates['date'] = pd.to_datetime(all_dates['date']).dt.strftime('%Y-%m-%d')
    daily_market_map = all_dates.groupby('date')['market_index'].mean().dropna().to_dict()
    print(f"  - Indexed daily market conditions for {len(daily_market_map)} calendar days.")

    # 3. Temporal Validation Split
    print(f"\n[2/6] Running Temporal Validation Experiment...")
    train_df_dated = train_df.copy()
    train_df_dated['date_dt'] = pd.to_datetime(train_df_dated['date'])
    train_df_dated['month'] = train_df_dated['date_dt'].dt.month

    # Months 1-8 for training, Months 9-10 for holdout validation
    tr_split = train_df_dated[train_df_dated['month'] <= 8].copy()
    va_split = train_df_dated[train_df_dated['month'] > 8].copy()

    val_model = FreightRateModel(lgb_weight=0.65, xgb_weight=0.35, random_state=42)
    val_model.daily_market_index_map.update(daily_market_map)
    val_metrics = val_model.fit(tr_split, val_df=va_split)

    print("  Holdout Validation Performance (Months 9-10):")
    print(f"    * Mean Absolute Error (MAE):     ${val_metrics['mae']:.2f}")
    print(f"    * Median Absolute Error (MedAE): ${val_metrics['medae']:.2f}")
    print(f"    * Root Mean Squared Error (RMSE): ${val_metrics['rmse']:.2f}")
    print(f"    * R-Squared (R²):                 {val_metrics['r2']:.4f}")
    print(f"    * Mean Absolute Percentage Error: {val_metrics['mape']:.2f}%")

    # 4. Train Full Production Model on Complete Labeled Development Data
    print(f"\n[3/6] Training Production Ensemble on Full Development Set ({len(train_df):,} loads)...")
    prod_model = FreightRateModel(lgb_weight=0.65, xgb_weight=0.35, random_state=42)
    prod_model.daily_market_index_map.update(daily_market_map)
    prod_model.fit(train_df)

    # Save model artifacts
    os.makedirs('models', exist_ok=True)
    prod_model.save('models/freight_rate_model.pkl')
    prod_model.save('model.pkl')
    print("  Saved production model artifacts to models/freight_rate_model.pkl and model.pkl")

    # 5. Generate Final Predictions for Validation Set
    print(f"\n[4/6] Generating predictions for validation set ({len(val_df):,} loads)...")
    val_preds = prod_model.predict(val_df)

    # Save submission file
    submission_df = pd.DataFrame({
        'load_id': val_df['load_id'],
        'predicted_rate': val_preds['predicted_rate']
    })
    submission_df.to_csv('validation_predictions.csv', index=False)
    if os.path.exists('data'):
        submission_df.to_csv('data/validation_predictions.csv', index=False)
    print(f"  Saved validation_predictions.csv ({len(submission_df):,} rows)")
    print(f"  Rate statistics: Mean=${submission_df['predicted_rate'].mean():.2f}, "
          f"Min=${submission_df['predicted_rate'].min():.2f}, Max=${submission_df['predicted_rate'].max():.2f}")

    # 6. Generate Predictions for December Fixed Inputs
    print(f"\n[5/6] Generating predictions for December chart inputs ({len(dec_df):,} rows)...")
    dec_features = dec_df.copy()
    dec_preds = prod_model.predict(dec_features)
    dec_df['predicted_rate'] = dec_preds['predicted_rate']

    # Keep original 7 columns
    out_cols = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    dec_out = dec_df[out_cols].copy()
    dec_out.to_csv('december-chart-inputs.csv', index=False)
    dec_out.to_csv('data/december_chart_inputs.csv', index=False)
    dec_out.to_csv('data/december-chart-inputs.csv', index=False)
    print(f"  Saved December predictions to data/december_chart_inputs.csv")
    print(f"  December rates: Mean=${dec_out['predicted_rate'].mean():.2f}, "
          f"Min=${dec_out['predicted_rate'].min():.2f}, Max=${dec_out['predicted_rate'].max():.2f}")

    # 7. Execute Scorer Validation
    print(f"\n[6/6] Executing score.py verification...")
    score_cmd = [
        sys.executable,
        'score.py',
        '--predictions', 'validation_predictions.csv',
        '--december-predictions', 'data/december_chart_inputs.csv'
    ]
    res = subprocess.run(score_cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print("ERROR:", res.stderr)
        sys.exit(res.returncode)

    print("=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == '__main__':
    main()
