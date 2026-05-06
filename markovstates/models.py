import numpy as np
import pandas as pd
import joblib
import os
from hmmlearn.hmm import GaussianHMM
from abc import ABC, abstractmethod

# did this with minimal help from claude, not even claude code. 
# All code here is typed by me, same goes for the rest of the files. 
# Just asked claude for advice/guidance in model training, OOP implementation

class WeatherModel(ABC):

    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        """Fit the model to a feature matrix."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return regime labels for each timestep."""
        pass

    @abstractmethod
    def score(self, X: np.ndarray) -> float:
        """Return a goodness-of-fit score."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass

class HMMWeatherModel(WeatherModel):

    def __init__(self, n_components: int, covar_type: str = 'diag',  n_restarts: int = 20):
        self.n_components = n_components
        self.covar_type = covar_type
        self.n_restarts = n_restarts
        self._model = None

    def fit(self, X: np.ndarray) -> None:
        """
        Fits a GaussianHMM model to feature matrix, then iterates through different
        seed values and scores each one, picking the best scored model
        """
        best_model, best_score = None, -np.inf

        for seed in range(self.n_restarts):
            model = GaussianHMM(
                n_components=self.n_components, 
                covariance_type=self.covar_type,
                n_iter=500,
                random_state=seed
            )
            model.fit(X)
            score = model.score(X)
            if score > best_score:
                best_model, best_score = model, score

        self._model = best_model
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        # returns regime labels for each 
        return self._model.predict(X)
    
    def transition_mat(self) -> np.ndarray:
        # returns transition matrix of model
        return self._model.transmat_

    def score(self, X: np.ndarray) -> float:
        return self._model.score(X)

    def bic(self, X: np.ndarray) -> float:
        return self._model.bic(X)
    
    def score_table(self, X: np.ndarray, n_range: tuple = (2,7)) -> pd.DataFrame:
        """
        return dataframe of AIC, BIC, and Seed scores for each respective model
        AIC = -2 × log-likelihood + 2 × n_parameters
        BIC = -2 × log-likelihood + n_parameters × log(n_samples)
        LL = Σ(n, i=1) ln(P(x_i | Θ))    , where each x_i is a state given our model parameters theta
        """
        results = pd.DataFrame(columns=["n_components", "log_likelihood", "AIC", "BIC"])

        st, en = n_range

        for n in range(st,en):
            for seed in range(self.n_restarts):
                model = GaussianHMM(
                    n_components=n,
                    covariance_type=self.covar_type,
                    n_iter=100,
                    random_state=seed
                )
                model.fit(X)
            new_row = pd.DataFrame([{
                        "n_components": n,
                        "log_likelihood": round(model.score(X), 4),
                        "AIC": round(model.aic(X), 4),
                        "BIC": round(model.bic(X), 4)
            }])
            results = pd.concat([results, new_row], ignore_index=True)
        
        return results

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self._model, path)

    def load(self, path: str) -> None:
        self._model = joblib.load(path)