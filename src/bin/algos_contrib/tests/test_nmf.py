from algos_contrib.NMF import NMF
from test.contrib_util import AlgoTestUtils


def test_algo():
    AlgoTestUtils.assert_algo_basic(NMF, serializable=False)
