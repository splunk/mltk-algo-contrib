from algos_contrib.MDS import MDS
from test.contrib_util import AlgoTestUtils


def test_algo():
    AlgoTestUtils.assert_algo_basic(MDS, serializable=False)
