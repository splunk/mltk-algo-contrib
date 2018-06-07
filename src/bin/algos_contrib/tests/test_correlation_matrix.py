from algos_contrib.CorrelationMatrix import CorrelationMatrix
from test.contrib_util import AlgoTestUtils


def test_algo():
    AlgoTestUtils.assert_algo_basic(CorrelationMatrix, serializable=False)
