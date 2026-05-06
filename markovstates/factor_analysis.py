from sklearn.decomposition import FactorAnalysis
import pandas as pd
import numpy as np
from markovstates.utils import Preprocess, hourly_dataframe

# constructing scaled feature matrix

# Add the RMT theory code here the whole time

pp = Preprocess(hourly_dataframe)
dfc_daily = pp.resample()
feature_cols = pp.extract_features(dfc_daily)
scaler = pp.fit_scaler(dfc_daily, feature_cols)
X = pp.apply_scaler(dfc_daily, feature_cols, scaler)

fa = FactorAnalysis(n_components=3, max_iter=3000, tol=1e-2, svd_method='lapack')
fa.fit(X)

loadings = pd.DataFrame(
    fa.components_.T,
    index=feature_cols,
    columns=["Factor 1", "Factor 2", "Factor 3"]
    )

if __name__ == "__main__":
    print(loadings.round(2))

FINAL_FEATURES = [
    "temperature_2m",
    "surface_pressure",
    "dew_point_2m"
]