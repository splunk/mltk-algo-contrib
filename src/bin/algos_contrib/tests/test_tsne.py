import pytest
from algos_contrib.TSNE import TSNE
from test.contrib_util import AlgoTestUtils

algo_options = {'feature_variables': ['Review']}


def test_algo():
    AlgoTestUtils.assert_algo_basic(TSNE, serializable=False)


def test_valid_params():
    algo_options['params'] = {'k': '1'}
    TSNE_algo = TSNE(algo_options)
    assert TSNE_algo.estimator.n_components == 1


def test_invalid_params_k_not_int():
    algo_options['params'] = {'k': '0.1'}
    with pytest.raises((RuntimeError, ValueError)) as excinfo:
        _ = TSNE(algo_options)
        assert excinfo.match('Invalid value for k: must be an int')


def test_invalid_params_k_not_valid():
    algo_options['params'] = {'k': '0'}
    with pytest.raises((RuntimeError, ValueError)) as excinfo:
        _ = TSNE(algo_options)
        assert excinfo.match('Invalid value for k: k must be greater than or equal to 1')


def test_default_parameter_values():
    algo_options['params'] = {'k': '1'}
    TSNE_algo = TSNE(algo_options)
    assert TSNE_algo.estimator.n_iter == 200
    assert TSNE_algo.estimator.perplexity == 30.0
    assert TSNE_algo.estimator.early_exaggeration == 4.0
    assert TSNE_algo.estimator.learning_rate == 100
