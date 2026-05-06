import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os
from markovstates.data_collect import hourly_dataframe

class Preprocess:

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def clean_df(self) -> pd.DataFrame:
        # remove the apparent temperature column to not have duplicate readings
        return self.df.drop(['apparent_temperature'], axis=1)

    def extract_features(self) -> list[str]:
        # return only numeric columns (exclude datetime objects)
        return self.df.select_dtypes(include=[np.number]).columns.tolist()

    def resample(self) -> pd.DataFrame:
        """
        Resamples the dataframe by days so repeated hourly correlations over 24h cycles
        don't screw up the model training, also removes hourly noise since 
        the time frame we're looking at is much longer than just hours.
        """
        dfc = self.df.set_index('date')
        dfc.index = pd.to_datetime(dfc.index)
        return dfc.resample('D').mean()

    def handle_missing(self, method: str = "interpolate") -> pd.DataFrame:
        """
        Fill missing values.

        Args:
            method: 'interpolate' (linear), 'ffill', or 'bfill'
        """
        if method == "interpolate":
            return self.df.interpolate(method="linear", limit_direction="both")
        elif method == "ffill":
            return self.df.ffill().bfill()
        elif method == "bfill":
            return self.df.bfill().ffill()
        else:
            raise ValueError(f"Unknown missing-value method: {method}")

    def fit_scaler(self, df: pd.DataFrame, feature_cols: list[str]) -> StandardScaler:
        # fit sklearn scaler to data
        scaler = StandardScaler()
        scaler.fit(df[feature_cols])
        return scaler

    def apply_scaler(self, df: pd.DataFrame, feature_cols: list[str], scaler: StandardScaler) -> np.ndarray:
        # apply trained scaler to dataframe, give us nice np array
        return scaler.transform(df[feature_cols]).astype(np.float64)

    # joblib methods to not lose scalers:

    def save_scaler(self, scaler: StandardScaler, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(scaler, path)

    def load_scaler(self, path: str) -> StandardScaler:
        return joblib.load(path)


class FeatMat(Preprocess):

    def __init__(self, df: pd.DataFrame, feats: list[str]) -> None:
        super().__init__(df)
        self.feats = feats

    def construct_feat_mat(self) -> np.ndarray:
        '''
        Cleans, resamples to daily, subsets to final features,
        scales, and returns the feature matrix ready for hmmlearn.
        '''
        cleaned = self.clean_df()
        self.df = cleaned
        daily = self.resample()
        daily = daily[self.feats]
        scaler = self.fit_scaler(daily, self.feats)
        X = self.apply_scaler(daily, self.feats, scaler)

        return X