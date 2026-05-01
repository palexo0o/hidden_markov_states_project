from sklearn.decomposition import FactorAnalysis
import pandas as pd
import numpy as np
from markovstates.preprocessing import Preprocess
from markovstates.data_collect import hourly_dataframe

# constructing variables

pp = Preprocess(hourly_dataframe)

dfc = pp.clean_df(hourly_dataframe)

feature_cols = pp.extract_features(dfc)

scaler = pp.fit_scaler(dfc, feature_cols)
feature_mat = pp.apply_scaler(dfc, feature_cols, scaler)
X = feature_mat

fa = FactorAnalysis(n_components=3)
fa.fit(X)

loadings = pd.DataFrame(
    fa.components_.T,
    index=feature_cols,
    columns=["Factor 1", "Factor 2", "Factor 3"]
)
print(loadings.round(2))

# after factor analysis, we find that the vars contributing the most real variance are:

FINAL_FEATURES = [
    "temperature_2m",      # Factor 2 anchor — thermal state
    "wind_speed_10m",      # Factor 1 anchor — kinetic state
    "wind_direction_10m",  # Factor 3 anchor — airmass origin
]

"""
Logic:



"""
