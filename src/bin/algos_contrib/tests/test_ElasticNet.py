#!/usr/bin/python
import pytest
from algos.ElasticNet import ElasticNet


def set_algo_params(args=None):
    algo_options = {
        'target_variable': ['petal_length'],
        'feature_variables': ['species', 'sepal_length'],
    }
    if args is not None:
        algo_options['params'] = args
    return algo_options


def make_valid_params():
    params = [
        {'alpha': '0.0'},
        {'alpha': '1'},
        {'alpha': '10'},
        {'l1_ratio': '0.1'},
        {'l1_ratio': '1'},
        {'fit_intercept': 't'},
        {'fit_intercept': 'f'},
        {'normalize': 't'},
        {'normalize': 'f'},
    ]
    return params


def make_invalid_params():
    params = [
        ({'alpha': '&*&d'}, 'Invalid value for alpha: must be a float'),
        ({'l1_ratio': '10'}, 'l1_ratio must be >= 0 and <= 1'),
        ({'l1_ratio': '&*&d'}, 'Invalid value for l1_ratio: must be a float'),
        ({'fit_intercept': 'nothing'}, 'Invalid value for fit_intercept: must be a boolean'),
        ({'normalize': 'nothing'}, 'Invalid value for normalize: must be a boolean'),
    ]
    return params


class TestMLPClassifier(object):

    # Sets up MLP Classifier and fits with default params to some target var
    @pytest.fixture(scope='function')
    def elastic_net_fit(self, iris):
        algo_options = set_algo_params()
        elastic_net = ElasticNet(algo_options)
        elastic_net.target_variable = algo_options['target_variable'][0]
        elastic_net.feature_variables = algo_options['feature_variables']
        elastic_net.fit(iris, algo_options)
        return elastic_net

    def test_fit(self, elastic_net_fit):
        assert len(elastic_net_fit.estimator.__dict__) == 17

    def test_predict(self, iris, elastic_net_fit):
        algo_options = set_algo_params()
        output_df = elastic_net_fit.apply(iris, options=algo_options)
        assert len(output_df.columns == 1)
        predicted_petal_length = output_df['predicted(petal_length)'].unique()
        assert len(predicted_petal_length) > 0

    # Tests with valid parameters
    @pytest.mark.parametrize('params', make_valid_params())
    def test_valid_params(self, params):
        algo_options = set_algo_params(params)
        ElasticNet(algo_options)

    # Tests with invalid parameters
    @pytest.mark.parametrize('params, error_msg', make_invalid_params())
    def test_invalid_params(self, params, error_msg):
        algo_options = set_algo_params(params)
        with pytest.raises((RuntimeError, ValueError)) as excinfo:
            ElasticNet(algo_options)
        assert excinfo.match(error_msg)
