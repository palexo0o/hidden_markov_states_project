from sklearn.decomposition import FactorAnalysis
import pandas as pd
import numpy as np
from preprocessing import fit_scaler, apply_scaler, extract_features
from data_collect import hourly_dataframe

# constructing variables

feature_cols = extract_features(hourly_dataframe)
scaler = fit_scaler(hourly_dataframe, feature_cols)
feature_mat = apply_scaler(hourly_dataframe, feature_cols, scaler)
X = feature_mat

fa = FactorAnalysis(n_components=3)
fa.fit(X)

loadings = pd.DataFrame(
    fa.components_.T,
    index=feature_cols,
    columns=["Factor 1", "Factor 2", "Factor 3"]
)
print(loadings.round(2))

# after factor analysis, we find that the vars contributing the most real variance are the ones below:

FINAL_FEATURES = [
    "apparent_temperature",   # Factor 1 anchor
    "wind_speed_10m",         # Factor 2 anchor
    "dew_point_2m",           # Factor 3 anchor
]