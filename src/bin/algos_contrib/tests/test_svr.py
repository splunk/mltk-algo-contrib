from algos_contrib.SVR import SVR
from test.contrib_util import AlgoTestUtils

import numpy as np
import pandas as pd


def test_algo_basic():
    input_df = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [4, 5, 6],
        'c': ['a', 'b', 'c'],
    })
    options = {
        'target_variable': ['a'],
        'feature_variables': ['b', 'c'],
    }
    required_methods = (
        '__init__',
        'fit',
        'partial_fit',
        'apply',
        'summary',
        'register_codecs',
    )
    AlgoTestUtils.assert_algo_basic(SVR, required_methods, input_df, options)


def test_prediction():
    training_df = pd.DataFrame({
        'y': [1, 2, 3],
        'x1': [4, 5, 6],
        'x2': [7, 8, 9],
    })
    options = {
        'target_variable': ['y'],
        'feature_variables': ['x1', 'x2'],
    }
    test_df = pd.DataFrame({
        'x1': [4],
        'x2': [7],
    })

    svr = SVR(options)
    svr.feature_variables = options['feature_variables']
    svr.target_variable = options['target_variable'][0]
    svr.fit(training_df, options)
    output = svr.apply(test_df, options)
    np.testing.assert_approx_equal(output['predicted(y)'].values, np.array([1.1]))

