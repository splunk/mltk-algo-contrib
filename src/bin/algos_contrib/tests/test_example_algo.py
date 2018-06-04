from algos_contrib.ExampleAlgo import ExampleAlgo
from test.contrib_util import AlgoTestUtils


def test_algo():
    AlgoTestUtils.assert_algo_basic(ExampleAlgo, serializable=False)

