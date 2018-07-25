from algos_contrib.CollaborativeFilter import CollaborativeFilter
from test.contrib_util import AlgoTestUtils


def test_algo():
    AlgoTestUtils.assert_algo_basic(CollaborativeFilter, serializable=False)
