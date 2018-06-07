from algos_contrib.SavgolFilter import SavgolFilter
from test.contrib_util import AlgoTestUtils


def test_algo():
    AlgoTestUtils.assert_algo_basic(SavgolFilter, serializable=False)
