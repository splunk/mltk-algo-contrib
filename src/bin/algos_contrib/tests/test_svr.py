from algos_contrib.SVR import SVR
from test.contrib_util import AlgoTestUtils

import pandas as pd


def test_algo():
    input_df = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [4, 5, 6],
        'c': ['a', 'b', 'c'],
    })
    options = {
        'target_variable': ['a'],
        'feature_variables': ['b', 'c'],
    }
    AlgoTestUtils.assert_algo_basic(SVR, input_df, options)

