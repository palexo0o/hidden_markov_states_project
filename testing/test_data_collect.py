import pytest
import pandas as pd
from markovstates.data_collect import hourly_dataframe, response


def test_dataframe_exists():
    """Verify that hourly_dataframe is created successfully."""
    assert hourly_dataframe is not None
    assert isinstance(hourly_dataframe, pd.DataFrame)


def test_dataframe_shape():
    """Check that dataframe has reasonable dimensions."""
    assert len(hourly_dataframe) > 0
    assert hourly_dataframe.shape[1] > 1


def test_date_column_exists():
    """Verify date column is present."""
    assert 'date' in hourly_dataframe.columns