import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from markovstates.utils import FeatMat, Preprocess, hourly_dataframe
from markovstates.factor_analysis import FINAL_FEATURES


def test_extract_features():
    """Test feature extraction (numeric columns only)."""
    pp = Preprocess(hourly_dataframe)
    features = pp.extract_features(hourly_dataframe)
    
    assert isinstance(features, list)
    assert len(features) > 0
    assert 'date' not in features
    for feat in features:
        assert hourly_dataframe[feat].dtype in [np.float64, np.float32, np.int64, np.int32]


def test_invalid_missing_method_raises():
    """Test that invalid method for handle_missing raises error."""
    pp = Preprocess(hourly_dataframe)
    with pytest.raises(ValueError):
        pp.handle_missing(method="invalid_method")


def test_scaled_features_normalized():
    """test that scaled features have mean ~0 and std ~1."""
    pp = Preprocess(hourly_dataframe)
    daily = pp.resample()
    features = FINAL_FEATURES
    
    scaler = pp.fit_scaler(daily, features)
    X_scaled = pp.apply_scaler(daily, features, scaler)
    
    means = X_scaled.mean(axis=0)
    assert np.allclose(means, 0, atol=0.01)
    stds = X_scaled.std(axis=0)
    assert np.allclose(stds, 1, atol=0.01)


def test_scaler_persistence():
    """test saving and loading da scaler."""
    pp = Preprocess(hourly_dataframe)
    daily = pp.resample()
    features = FINAL_FEATURES
    
    scaler = pp.fit_scaler(daily, features)
    test_path = '/tmp/test_scaler.pkl'
    
    pp.save_scaler(scaler, test_path)
    assert np.isfile(test_path)
    
    loaded_scaler = pp.load_scaler(test_path)
    assert isinstance(loaded_scaler, StandardScaler)
    assert np.allclose(scaler.mean_, loaded_scaler.mean_)
    assert np.allclose(scaler.scale_, loaded_scaler.scale_)


def test_featmat_initialization():
    """test featmat class initialization."""
    fm = FeatMat(hourly_dataframe, FINAL_FEATURES)
    assert fm is not None
    assert fm.feats == FINAL_FEATURES

