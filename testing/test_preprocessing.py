import pytest
import numpy as np
from markovstates.utils import FINAL_FEATURES, hourly_dataframe, FeatMat, Preprocess

FM = FeatMat(hourly_dataframe, FINAL_FEATURES)

Xnew = FM.construct_feat_mat()

print(Xnew)
print(Xnew.shape) # 768 time steps over 3 factors; cool beans cool beans that's a whole lotta reefer lemme help u wit da preroll