#!/usr/bin/env python

import pytest
import pandas as pd
import numpy as np

from sklearn.feature_selection import GenericUnivariateSelect as GUS

from algos.FieldSelector import FieldSelector
from algos.FieldSelector import has_required_version


def make_valid_params():
    return [
        ('k_best', '5'),
        ('percentile', '5'),
        ('fdr', '0.1'),
        ('fpr', '0.2'),
        ('fwe', '0.3'),
        ('fpr', '0.99'),
    ]


def make_invalid_params():
    return [
        ('k_best', '5.1'),
        ('percentile', '123.123'),
        ('fpr', '1'),
        ('fdr', '2'),
        ('fwe', '-1'),
        ('fpr', '0'),
    ]


@pytest.fixture(scope='function')
def with_raise_on_invalid_version():
    if has_required_version():
        yield
    else:
        with pytest.raises((RuntimeError, ImportError)):
            yield


class TestFieldSelector(object):
    @pytest.fixture(scope='function')
    def options(self):
        return {
            'target_variable': ['species'],
            'feature_variables': ['petal_length', 'petal_width'],
        }

    @pytest.fixture(scope='function')
    def create_data(self):
        df = pd.DataFrame(
            {
                'petal_length': [1, 5, 3, np.nan, 5.6, 4.2],
                'petal_width': [np.nan, 4, 2, 6.4, 2.6, 1.2],
                'species': [
                    'iris_setosa',
                    'iris_setosa',
                    'iris_versicolor',
                    'iris_virginica',
                    'iris_virginica',
                    'iris_virginica',
                ],
            }
        )
        return df

    @pytest.fixture(scope='function')
    def fs_instance(self, options, create_data):
        field_selector = FieldSelector(options)
        field_selector.feature_variables = options.get('feature_variables', [])
        field_selector.target_variable = options.get('target_variable', [])[0]
        field_selector.fit(create_data, options)
        return field_selector

    @pytest.mark.parametrize('mode,param', make_valid_params())
    def test_init_valid_mode_param(self, options, mode, param):
        options.update({'params': {'param': param, 'mode': mode}})
        field_selector = FieldSelector(options)
        assert isinstance(field_selector.estimator, GUS)

        if mode in ['k_best', 'percentile']:
            assert isinstance(field_selector.estimator.param, int)

    @pytest.mark.parametrize('mode,param', make_invalid_params())
    def test_init_invalid_mode_param(self, options, mode, param):
        options.update({'params': {'param': param, 'mode': mode}})
        with pytest.raises(ValueError):
            FieldSelector(options)

    @pytest.mark.parametrize('param', [('mode'), ('type')])
    def test_invalid_mode_type(self, options, param):
        options.update({'params': {param: 'fake'}})
        with pytest.raises(RuntimeError):
            FieldSelector(options)

    @pytest.mark.parametrize('score_func', [('numeric'), ('numerical'), ('categorical')])
    def test_valid_score_func(self, options, score_func):
        options.update({'params': {'type': score_func}})
        field_selector = FieldSelector(options)
        assert isinstance(field_selector.estimator, GUS)

    @pytest.mark.usefixtures('with_raise_on_invalid_version')
    def test_summary_columns(self, fs_instance):
        summary_options = {'model_name': 'fake', 'mlspl_limits': {}}
        summary = fs_instance.summary(summary_options)
        assert set(summary.columns) == {'feature_variables', 'score', 'p-value'}

    @pytest.mark.usefixtures('with_raise_on_invalid_version')
    def test_summary_invalid_options(self, fs_instance):
        summary_options = {'model_name': 'fake', 'mlspl_limits': {}, 'scores': 'true'}
        with pytest.raises(RuntimeError, match=r'models do not take options for summarization'):
            fs_instance.summary(summary_options)
