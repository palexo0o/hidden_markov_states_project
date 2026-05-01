import pytest

from markovstates.utils import hourly_dataframe


print(hourly_dataframe)
print(hourly_dataframe.isna().sum())