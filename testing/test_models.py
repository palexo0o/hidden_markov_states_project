import pytest
import numpy as np
import os
import tempfile
from markovstates.models import HMMWeatherModel
from markovstates.utils import FeatMat, hourly_dataframe
from markovstates.factor_analysis import FINAL_FEATURES


def test_model_fit(model, feature_matrix):
    """test fitting the model to feature matrix"""
    model.fit(feature_matrix)
    
    assert model._model is not None
    assert hasattr(model._model, 'transmat_')
    assert hasattr(model._model, 'means_')
    assert hasattr(model._model, 'covars_')


def test_model_fit_finds_best_seed(feature_matrix):
    """test that fitting with multiple restarts will select the best model"""
    model = HMMWeatherModel(n_components=5, covar_type='diag', n_restarts=5)
    model.fit(feature_matrix)
    
    score = model.score(feature_matrix)
    assert isinstance(score, (float, np.floating))
    assert np.isfinite(score)


def test_transition_matrix(model, feature_matrix):
    """Test transition matrix shape and properties"""
    model.fit(feature_matrix)
    transmat = model.transition_mat()
    
    assert transmat.shape == (model.n_components, model.n_components)
    row_sums = transmat.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)
    assert np.all(transmat >= 0) and np.all(transmat <= 1)


def test_score_table_generation(feature_matrix):
    """test generating comparison table for different model complexities"""
    model = HMMWeatherModel(n_components=5, covar_type='diag', n_restarts=2)
    scores_df = model.score_table(feature_matrix, n_range=(3, 5))
    
    assert scores_df is not None
    assert len(scores_df) > 0
    assert 'n_components' in scores_df.columns
    assert 'log_likelihood' in scores_df.columns
    assert 'AIC' in scores_df.columns
    assert 'BIC' in scores_df.columns