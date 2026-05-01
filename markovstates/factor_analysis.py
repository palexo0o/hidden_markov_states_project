from sklearn.decomposition import FactorAnalysis
import pandas as pd
import numpy as np
from markovstates.utils import Preprocess, hourly_dataframe

# constructing variables

pp = Preprocess(hourly_dataframe)
dfc = pp.clean_df(hourly_dataframe)
dfc = dfc.set_index('date')
dfc.index = pd.to_datetime(dfc.index)
dfc_daily = dfc.resample('D').mean()
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
    "temperature_2m",      # Factor 2 anchor — thermal state
    "wind_speed_10m",      # Factor 1 anchor — kinetic state
    "wind_direction_10m",  # Factor 3 anchor — airmass origin
]

"""
Logic behind feature selection:

                      Factor 1  Factor 2  Factor 3
temperature_2m            0.32      0.72      0.28
relative_humidity_2m     -0.64     -0.77     -0.00
dew_point_2m             -0.55     -0.38      0.35
precipitation            -0.19     -0.14      0.06
surface_pressure          0.21      0.13     -0.25
cloud_cover              -0.15     -0.13      0.14
cloud_cover_mid          -0.20     -0.21      0.26
wind_speed_10m            0.92     -0.29      0.01
wind_speed_100m           0.91     -0.41      0.01
wind_direction_10m       -0.31      0.03      0.80
wind_direction_100m      -0.28      0.06      0.85
direct_radiation          0.16      0.61     -0.06

if we look at the above results from the Factor Analysis on our weather data,
we find the following

For underlying factor 1, we can see that wind speed has drastically the largest
impact out of all the variables, implying that Factor 1is most probably 
some kind of wind strength axis. Hence, we choose wind_speed_10m as our
Factor 1 anchor. 

For underlying Factor 2, we see that radiation, temperature, dew point, and 
humidity have the highest values, implying that Factor 2 is some kind of 
thermal/moisture axis. We shall opt for temperature_2m since dew point and humidity
variables already factor in temperature in their calculations, hence why they probably
don't register as strong as temperature by itself in terms of contributing new 
information. 

For underlying Factor 3, wind direction has by far the strongest 


"""
