import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os
from markovstates.data_collect import hourly_dataframe

# After a hiccup with factor analysis, we must drop one of the temperature columns as it was overcorrelating with
# one of the other temperature values

class Preprocess:

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    # define instance methods for preprocessing object

    def clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # remove the apparent_temperature column
        return df.drop(['apparent_temperature'], axis=1)

    def extract_features(self, df: pd.DataFrame) -> list[str]:
        # Return only numeric columns (exclude date/timestamp columns, and remove apparent_temp column)
        return df.select_dtypes(include=[np.number]).columns.tolist()

    def handle_missing(self, df: pd.DataFrame, method: str = "interpolate") -> pd.DataFrame:
        """
        Fill missing values.

        Args:
            method: 'interpolate' (linear), 'ffill', or 'bfill'
        """

        # REMEMBER TO IMPLEMENT ERROR HANDLING FOR THIS FUNCTION (TRY/EXCEPT)
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

    def fit_scaler(self, df: pd.DataFrame, feature_cols: list[str]) -> StandardScaler:
        scaler = StandardScaler()
        scaler.fit(df[feature_cols])
        return scaler

    def apply_scaler(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        scaler: StandardScaler,
        ) -> np.ndarray:
        """Returns a (timesteps × n_features) float64 array ready for hmmlearn."""
        return scaler.transform(df[feature_cols]).astype(np.float64)

    def save_scaler(self, scaler: StandardScaler, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(scaler, path)

    def load_scaler(self, path: str) -> StandardScaler:
        return joblib.load(path)

if __name__ == "__main__":
    from markovstates.data_collect import hourly_dataframe
    
    pp = Preprocess(hourly_dataframe)

    feature_cols = pp.extract_features(hourly_dataframe)
    scaler = pp.fit_scaler(hourly_dataframe, feature_cols)
    pp.save_scaler(scaler, 'models/scaler.pkl')
    feature_mat = pp.apply_scaler(hourly_dataframe, feature_cols, scaler)


'''
After factor analysis we find that the below features are most valuable. Imported as FINAL_FEATURES from 
factor_analysis.py and factor_explore.ipynb

We shall now fit and scale the updated features matrix with these new key components,
I will do so through a class again with our updated scaled feature matrix
and relevant operations with it as instance methods:
'''

class FeatMat(Preprocess):

    def __init__(self, df: pd.DataFrame, feats: list[str]) -> None:
        super().__init__(df)
        self.feats = feats

    def construct_feat_mat(self) -> np.ndarray:
        '''
        this function retrieves the optimal features from the FA file, then
        constructs a new df with just those features, then applies the preprocessing
        techniques for scaling, cleaning, and normalizing our matrix for our models
        '''

        UPDATED_DF = self.df.loc[:, self.feats]
        sclr = StandardScaler()
        sclr.fit(UPDATED_DF)
        X = sclr.transform(UPDATED_DF).astype(np.float64)

        return X
