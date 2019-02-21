from algos_contrib.IsolationForest import IsolationForest
from test.contrib_util import AlgoTestUtils
import pandas as pd

def test_algo():
    AlgoTestUtils.assert_algo_basic(IsolationForest, serializable=False)

def test_algo_options():
    input_df = pd.DataFrame({
        'a': [5.1, 4.9, 4.7, 4.6],
        'b': [3.5, 3.0, 3.1, 3.2],
        'c': [1.4, 1.4, 1.5, 1.6],
        'd': [0.2, 0.2, 0.2, 0.4],
        'e': ['Iris Setosa','Iris Setosa','Iris Versicolor','Iris Virginica']
    })
    options = {
        'target_variables' : [],
        'feature_variables': ['a','b','c','d'],
    }
    required_methods = (
        '__init__',
        'fit',
        'apply',
        'register_codecs',
    )
    AlgoTestUtils.assert_algo_basic(IsolationForest, required_methods=required_methods, input_df=input_df, options=options, serializable=False)