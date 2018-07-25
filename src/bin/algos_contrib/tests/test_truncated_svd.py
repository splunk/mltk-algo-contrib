import pandas as pd
from algos_contrib.TruncatedSVD import TruncatedSVD
from test.contrib_util import AlgoTestUtils


def test_algo():
    input_df = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [4, 5, 6],
        'c': ['a', 'b', 'c'],
    })
    options = {
        'feature_variables': ['a', 'b', 'c'],
    }
    required_methods = (
        '__init__',
        'fit',
        'partial_fit',
        'apply',
        'summary',
        'register_codecs',
    )
    AlgoTestUtils.assert_algo_basic(TruncatedSVD, required_methods, input_df, options)
