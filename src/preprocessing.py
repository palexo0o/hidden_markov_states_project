import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os
from data_collect import hourly_dataframe
from factor_analysis import FINAL_FEATURES

def extract_features(df: pd.DataFrame) -> list[str]:
    # Return only numeric columns (exclude date/timestamp columns)
    return df.select_dtypes(include=[np.number]).columns.tolist()

def handle_missing(df: pd.DataFrame, method: str = "interpolate") -> pd.DataFrame:
    """
    Fill missing values.

    Args:
        method: 'interpolate' (linear), 'ffill', or 'bfill'
    """
    if method == "interpolate":
        return df.interpolate(method="linear", limit_direction="both")
    elif method == "ffill":
        return df.ffill().bfill()   # bfill catches any leading NaNs
    elif method == "bfill":
        return df.bfill().ffill()
    else:
        raise ValueError(f"Unknown missing-value method: {method}")
    
# the nice thing here luckily is that openmeteo is a great source and we have literally zero null values
# but lets just keep it for any situation where we do actually get a dataframe with nulls

# scaling operations:

def fit_scaler(df: pd.DataFrame, feature_cols: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df[feature_cols])
    return scaler

def apply_scaler(
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler: StandardScaler,
) -> np.ndarray:
    """Returns a (timesteps × n_features) float64 array ready for hmmlearn."""
    return scaler.transform(df[feature_cols]).astype(np.float64)

def save_scaler(scaler: StandardScaler, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)

def load_scaler(path: str) -> StandardScaler:
    return joblib.load(path)

if __name__ == "__main__":
    from data_collect import hourly_dataframe
    
    feature_cols = extract_features(hourly_dataframe)
    scaler = fit_scaler(hourly_dataframe, feature_cols)
    save_scaler(scaler, 'models/scaler.pkl')
    feature_mat = apply_scaler(hourly_dataframe, feature_cols, scaler)


'''
Sfter factor analysis we find that the below features are most valuable. Imported as FINAL_FEATURES

We shall now fit and scale the updated features matrix with these new key components:
'''

clean_df = hourly_dataframe['apparent_temperature', ]