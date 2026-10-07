"""
Feature Engineering and Data Preprocessing Pipeline for Spotter Freight Rate Prediction.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


# Known coordinates for all 72 freight market cities
KNOWN_CITIES: Dict[str, Tuple[float, float]] = {
    'Albany': (42.65258, -73.75623),
    'Albuquerque': (35.08449, -106.65114),
    'Allentown': (40.01684, -73.80889),
    'Amarillo': (35.22199, -101.8313),
    'Atlanta': (34.84933, -86.2894),
    'Austin': (30.26715, -97.74306),
    'Bakersfield': (33.28832, -116.67935),
    'Baltimore': (38.16908, -72.74564),
    'Baton Rouge': (30.50134, -92.65374),
    'Birmingham': (33.52066, -86.80249),
    'Boston': (42.36008, -71.05888),
    'Buffalo': (42.88645, -78.87837),
    'Charlotte': (34.65272, -81.47202),
    'Chicago': (40.512, -86.86737),
    'Cincinnati': (39.10312, -84.51202),
    'Cleveland': (41.49932, -81.69436),
    'Columbia': (34.6357, -83.28506),
    'Columbus': (39.96118, -82.99879),
    'Corpus Christi': (28.61374, -95.89569),
    'Dallas': (31.83025, -94.38343),
    'Dayton': (39.87206, -85.62931),
    'Denver': (39.73924, -104.99025),
    'Des Moines': (41.58684, -93.62496),
    'Detroit': (43.61923, -84.04517),
    'El Paso': (31.76188, -106.48502),
    'Fort Wayne': (41.31561, -85.36206),
    'Fresno': (33.07921, -119.79528),
    'Grand Rapids': (42.96336, -85.66809),
    'Green Bay': (44.30296, -87.52871),
    'Harrisburg': (39.71695, -78.12481),
    'Hartford': (39.55328, -72.18051),
    'Houston': (29.76043, -95.3698),
    'Indianapolis': (39.7684, -86.15807),
    'Jackson': (32.07771, -90.92587),
    'Jacksonville': (30.33218, -81.65565),
    'Kansas City': (39.41104, -91.1245),
    'Knoxville': (36.25725, -86.02539),
    'Laredo': (25.5, -97.22681),
    'Las Vegas': (36.16994, -115.13983),
    'Lexington': (36.99152, -84.99876),
    'Little Rock': (34.74648, -92.28959),
    'Los Angeles': (28.56624, -116.70249),
    'Louisville': (38.25266, -85.75846),
    'Memphis': (35.56802, -89.52871),
    'Miami': (25.76168, -80.19179),
    'Milwaukee': (43.0389, -87.90647),
    'Minneapolis': (44.97775, -93.26501),
    'Mobile': (31.80442, -88.25195),
    'Montgomery': (31.37602, -86.10743),
    'Nashville': (35.29479, -88.08915),
    'New Orleans': (29.95107, -90.07153),
    'New York': (40.71278, -74.00594),
    'Norfolk': (37.12278, -75.87106),
    'Oklahoma City': (35.46756, -97.51643),
    'Omaha': (41.25654, -95.9345),
    'Orlando': (28.53834, -81.37924),
    'Philadelphia': (39.22317, -72.9671),
    'Phoenix': (28.75668, -115.87725),
    'Pittsburgh': (40.44062, -79.99589),
    'Portland': (44.30296, -121.69849),
    'Raleigh': (35.37376, -77.47605),
    'Richmond': (38.09122, -76.78906),
    'Sacramento': (38.58157, -121.4944),
    'Salt Lake City': (40.76078, -111.89105),
    'San Antonio': (29.50969, -98.40059),
    'San Diego': (32.00929, -116.89598),
    'San Francisco': (35.1967, -121.69849),
    'Savannah': (33.60973, -82.29994),
    'Seattle': (44.30296, -121.69849),
    'St. Louis': (38.627, -90.1994),
    'Syracuse': (43.04812, -76.14742),
    'Tampa': (27.95058, -82.45718),
}

# Equipment typical median weights (in lbs)
EQUIPMENT_MEDIAN_WEIGHT: Dict[str, float] = {
    'Dry Van': 31200.0,
    'Reefer': 32400.0,
    'Flatbed': 30800.0,
}

# Equipment base rate per mile adjustments
EQUIPMENT_BASE_RPM: Dict[str, float] = {
    'Dry Van': 2.05,
    'Reefer': 2.08,
    'Flatbed': 2.08,
}


def haversine_distance(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Computes great-circle distance in miles between coordinates."""
    r_lat1, r_lon1 = np.radians(lat1), np.radians(lon1)
    r_lat2, r_lon2 = np.radians(lat2), np.radians(lon2)
    dlat = r_lat2 - r_lat1
    dlon = r_lon2 - r_lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(r_lat1) * np.cos(r_lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return 3958.8 * c


def clean_dataframe(
    df: pd.DataFrame,
    daily_market_index_map: Optional[Dict[str, float]] = None,
    lane_quote_signal_map: Optional[Dict[Tuple[str, str, str], float]] = None,
) -> pd.DataFrame:
    """
    Cleans raw freight dataframe:
    - Fixes negative weights using abs()
    - Imputes missing weights using equipment medians
    - Resolves missing lat/lon coordinates from city registry
    - Imputes missing market_index and quote_signal intelligently
    """
    cleaned = df.copy()

    # Convert date to datetime
    if 'date' in cleaned.columns:
        cleaned['date'] = pd.to_datetime(cleaned['date'])

    # 1. Coordinate lookup if missing
    for col_prefix, city_col in [('pickup', 'pickup'), ('delivery', 'delivery')]:
        lat_col = f'{col_prefix}_lat'
        lon_col = f'{col_prefix}_lon'
        if lat_col not in cleaned.columns or cleaned[lat_col].isna().any():
            cleaned[lat_col] = cleaned[lat_col] if lat_col in cleaned.columns else np.nan
            cleaned[lon_col] = cleaned[lon_col] if lon_col in cleaned.columns else np.nan
            for idx, row in cleaned.iterrows():
                city = row[city_col]
                if pd.isna(cleaned.at[idx, lat_col]) and city in KNOWN_CITIES:
                    cleaned.at[idx, lat_col] = KNOWN_CITIES[city][0]
                    cleaned.at[idx, lon_col] = KNOWN_CITIES[city][1]

    # 2. Fix weights: abs() and median imputation
    if 'weight' in cleaned.columns:
        cleaned['weight'] = cleaned['weight'].abs()
        for eq, med in EQUIPMENT_MEDIAN_WEIGHT.items():
            mask = cleaned['equipment'].eq(eq) & cleaned['weight'].isna()
            cleaned.loc[mask, 'weight'] = med
        cleaned['weight'] = cleaned['weight'].fillna(31500.0)
    else:
        cleaned['weight'] = 31500.0

    # 3. Market index imputation
    if 'market_index' not in cleaned.columns:
        cleaned['market_index'] = np.nan

    # First from daily means if date exists and map provided
    if daily_market_index_map is not None and 'date' in cleaned.columns:
        date_str_series = cleaned['date'].dt.strftime('%Y-%m-%d')
        cleaned['market_index'] = cleaned['market_index'].fillna(date_str_series.map(daily_market_index_map))

    # Then from in-data daily mean
    if 'date' in cleaned.columns and cleaned['market_index'].isna().any():
        daily_mean = cleaned.groupby('date')['market_index'].transform('mean')
        cleaned['market_index'] = cleaned['market_index'].fillna(daily_mean)

    # Global default fallback
    cleaned['market_index'] = cleaned['market_index'].fillna(1.0)

    # 4. Quote signal imputation
    if 'quote_signal' not in cleaned.columns:
        cleaned['quote_signal'] = np.nan

    if lane_quote_signal_map is not None:
        lane_keys = list(zip(cleaned['pickup'], cleaned['delivery'], cleaned['equipment']))
        mapped_signals = [lane_quote_signal_map.get(k, np.nan) for k in lane_keys]
        cleaned['quote_signal'] = cleaned['quote_signal'].fillna(pd.Series(mapped_signals, index=cleaned.index))

    # Equipment default fallback
    for eq, base_sig in EQUIPMENT_BASE_RPM.items():
        mask = cleaned['equipment'].eq(eq) & cleaned['quote_signal'].isna()
        cleaned.loc[mask, 'quote_signal'] = base_sig
    cleaned['quote_signal'] = cleaned['quote_signal'].fillna(2.05)

    return cleaned


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes spatial, operational, cyclical, and market interaction features.
    """
    feat = df.copy()

    # Coordinates
    lat1 = feat['pickup_lat'].to_numpy(dtype=float)
    lon1 = feat['pickup_lon'].to_numpy(dtype=float)
    lat2 = feat['delivery_lat'].to_numpy(dtype=float)
    lon2 = feat['delivery_lon'].to_numpy(dtype=float)

    # Geospatial metrics
    feat['haversine_dist'] = haversine_distance(lat1, lon1, lat2, lon2)
    feat['circuity'] = feat['distance'] / (feat['haversine_dist'] + 1.0)
    feat['delta_lat'] = feat['delivery_lat'] - feat['pickup_lat']
    feat['delta_lon'] = feat['delivery_lon'] - feat['pickup_lon']
    feat['mid_lat'] = (feat['delivery_lat'] + feat['pickup_lat']) / 2.0
    feat['mid_lon'] = (feat['delivery_lon'] + feat['pickup_lon']) / 2.0
    feat['bearing'] = np.arctan2(feat['delta_lon'], feat['delta_lat'])

    # Freight operational features
    feat['weight_per_mile'] = feat['weight'] / (feat['distance'] + 1.0)
    feat['is_heavy'] = (feat['weight'] >= 40000.0).astype(int)
    feat['is_short_haul'] = (feat['distance'] <= 350.0).astype(int)
    feat['is_long_haul'] = (feat['distance'] >= 1500.0).astype(int)

    # Temporal & Calendar features
    dates = pd.to_datetime(feat['date'])
    feat['month'] = dates.dt.month
    feat['day'] = dates.dt.day
    feat['dayofweek'] = dates.dt.dayofweek
    feat['dayofyear'] = dates.dt.dayofyear
    feat['is_weekend'] = (feat['dayofweek'] >= 5).astype(int)
    feat['is_month_end'] = (feat['day'] >= 25).astype(int)

    # Cyclical encodings
    feat['sin_doy'] = np.sin(2.0 * np.pi * feat['dayofyear'] / 365.25)
    feat['cos_doy'] = np.cos(2.0 * np.pi * feat['dayofyear'] / 365.25)
    feat['sin_dow'] = np.sin(2.0 * np.pi * feat['dayofweek'] / 7.0)
    feat['cos_dow'] = np.cos(2.0 * np.pi * feat['dayofweek'] / 7.0)

    # Market signals & interactions
    feat['expected_cost'] = feat['distance'] * feat['quote_signal']
    feat['expected_cost_market'] = feat['expected_cost'] * feat['market_index']
    feat['market_adjusted_quote'] = feat['quote_signal'] * feat['market_index']
    feat['signal_spread'] = feat['market_adjusted_quote'] - feat['quote_signal']

    return feat
