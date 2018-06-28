from algos_contrib.TSNE import TSNE
from test.contrib_util import AlgoTestUtils


def test_algo():
    AlgoTestUtils.assert_algo_basic(TSNE, serializable=False)
