from algos_contrib.LatentDirichletAllocation import LatentDirichletAllocation
from test.contrib_util import AlgoTestUtils


def test_algo():
    AlgoTestUtils.assert_algo_basic(LatentDirichletAllocation, serializable=False)
