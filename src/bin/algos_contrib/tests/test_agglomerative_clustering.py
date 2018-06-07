from algos_contrib.AgglomerativeClustering import AgglomerativeClustering
from test.contrib_util import AlgoTestUtils


def test_algo():
    AlgoTestUtils.assert_algo_basic(AgglomerativeClustering, serializable=False)
