from algos_contrib.TruncatedSVD import TruncatedSVD
from test.contrib_util import AlgoTestUtils


def test_algo():
    AlgoTestUtils.assert_algo_basic(TruncatedSVD, serializable=False)
