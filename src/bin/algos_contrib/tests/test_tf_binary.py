from algos_contrib.TFBinary import TFBinary
from test.contrib_util import AlgoTestUtils


def test_algo():
    AlgoTestUtils.assert_algo_basic(TFBinary, serializable=False)
