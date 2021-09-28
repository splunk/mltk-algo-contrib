#!/usr/bin/env python

from pytest import raises
import pandas as pd
import numpy as np
import mock

from algos.ACF import ACF
from algos.PACF import PACF


class TestACF(object):
    """ACF Test Class. PACF inherits from ACF so PACF is smaller."""

    @staticmethod
    def setup(df=None, options=None, acf=True):
        """Pass df and options to an algo."""
        if acf:
            algo = ACF
        else:
            algo = PACF

        if options is not None:
            algo = algo(options)
            algo.feature_variables = options.get('feature_variables')
            return algo.fit(df, options)

    def test_init_valid(self):
        algo_options = {'feature_variables': ['Voltage']}
        ACF(algo_options)
        # no error

    def test_init_error(self):
        algo_options = {'feature_variables': ['Voltage'], 'target_variable': ['anything']}

        # no target variables allowed
        with raises(RuntimeError):
            ACF(algo_options)

    def test_init_no_fields(self):
        algo_options = {'feature_variables': []}

        # no target variables allowed
        with raises(RuntimeError):
            ACF(algo_options)

    def test_k(self, ts_voltage):
        algo_options = {'feature_variables': ['Voltage'], 'params': {'k': '100'}}
        for is_acf in [True, False]:
            df = self.setup(ts_voltage, algo_options, acf=is_acf)
            assert len(df.columns) == 4
            assert len(df) == 101

    def test_conf_interval(self, ts_voltage):
        algo_options = {
            'feature_variables': ['Voltage'],
            'params': {'conf_interval': '99', 'k': '100'},
        }
        for is_acf in [True, False]:
            df = self.setup(ts_voltage, algo_options, acf=is_acf)
            assert len(df.columns) == 4
            assert len(df) == 101
            assert 'lower99(%s(Voltage))' % ('acf' if is_acf else 'pacf') in df.columns
            assert 'upper99(%s(Voltage))' % ('acf' if is_acf else 'pacf') in df.columns

    def test_fft(self, ts_voltage):
        algo_options = {'feature_variables': ['Voltage'], 'params': {'fft': '1', 'k': '100'}}
        df = self.setup(ts_voltage, algo_options)
        assert len(df.columns) == 4
        assert len(df) == 101

    def test_PACF_method(self, ts_voltage):
        algo_options = {'feature_variables': ['Voltage'], 'params': {'k': '100'}}
        for method in ['ywunbiased', 'ywmle', 'ols']:
            algo_options['params']['method'] = method
            self.setup(ts_voltage, algo_options, False)

    def validate_error_by_param(
        self, algo_options, df, param_key, value_error_tuple_list, is_acf=True
    ):
        for (value, error) in value_error_tuple_list:
            # update algo_options to use a different value for specified param per iteration
            if 'params' in algo_options:
                algo_options['params'][param_key] = value
            else:
                algo_options['params'] = {param_key: value}
            with raises((RuntimeError, ValueError)) as e:
                # pass in algo_options with invalid param value and expect an exception to be raised
                self.setup(df, algo_options, acf=is_acf)

    def test_k_invalid(self, ts_voltage):
        algo_options = {'feature_variables': ['Voltage']}
        param_name = 'k'
        acf_k_value_error_tuple_list = [
            ('0', 'k must be greater than 0.'),
            ('999', 'k must be less than number of events'),
            ('10000', 'k must be less than number of events'),
            ('Foo', 'must be an int'),
            ('1.5', 'must be an int'),
            ('-1', 'k must be greater than 0'),
            ('-10', 'k must be greater than 0'),
        ]
        self.validate_error_by_param(
            algo_options, ts_voltage, param_name, acf_k_value_error_tuple_list
        )
        pacf_k_value_error_tuple_list = [
            ('999', 'k must be less than number of events'),
            ('10000', 'k must be less than number of events'),
            ('Foo', 'must be an int'),
            ('1.5', 'must be an int'),
        ]
        self.validate_error_by_param(
            algo_options, ts_voltage, param_name, pacf_k_value_error_tuple_list, is_acf=False
        )

    def test_conf_invalid(self, ts_voltage):
        algo_options = {'feature_variables': ['Voltage']}
        param_name = 'conf_interval'
        acf_ci_value_error_tuple_list = [
            ('0', 'conf_interval cannot be less than 1 or more than 99'),
            ('100', 'conf_interval cannot be less than 1 or more than 99'),
            ('-10', 'conf_interval cannot be less than 1 or more than 99'),
            ('95.5', 'must be an int'),
            ('Foo', 'must be an int'),
        ]
        self.validate_error_by_param(
            algo_options, ts_voltage, param_name, acf_ci_value_error_tuple_list
        )
        pacf_ci_value_error_tuple_list = [
            ('0', 'conf_interval cannot be less than 1 or more than 99'),
            ('100', 'conf_interval cannot be less than 1 or more than 99'),
            ('-10', 'conf_interval cannot be less than 1 or more than 99'),
            ('Foo', 'must be an int'),
        ]
        self.validate_error_by_param(
            algo_options, ts_voltage, param_name, pacf_ci_value_error_tuple_list, is_acf=False
        )

    def test_fft_invalid(self, ts_voltage):
        algo_options = {'feature_variables': ['Voltage']}
        param_name = 'fft'
        fft_value_error_tuple_list = [('2', 'must be a boolean'), ('Foo', 'must be a boolean')]
        self.validate_error_by_param(
            algo_options, ts_voltage, param_name, fft_value_error_tuple_list
        )

    def test_method_invalid(self, ts_voltage):
        algo_options = {'feature_variables': ['Voltage']}
        param_name = 'method'
        method_value_error_tuple_list = [
            ('', 'must be a non-empty string'),
            ('Foo', 'method not available'),
        ]
        self.validate_error_by_param(
            algo_options, ts_voltage, param_name, method_value_error_tuple_list, is_acf=False
        )

    def test_no_events(self):
        df = pd.DataFrame()
        expected_error = 'No valid events'
        algo_options = {'feature_variables': ['Voltage']}
        for is_acf in [True, False]:
            with raises(RuntimeError) as e:
                self.setup(df, algo_options, acf=is_acf)
            assert expected_error in str(e.value)

    def test_field_not_in_df(self, iris):
        algo_options = {'feature_variables': ['Foo']}
        expected_error = 'No valid fields'
        for is_acf in [True, False]:
            with raises(RuntimeError) as e:
                self.setup(iris, algo_options, acf=is_acf)
            assert expected_error in str(e.value)

    def test_multiple_fields_error(self, iris):
        algo_options = {'feature_variables': ['petal_length', 'petal_width']}
        expected_error = 'You must specify one field'
        for is_acf in [True, False]:
            with raises(RuntimeError) as e:
                self.setup(iris, algo_options, acf=is_acf)
            assert expected_error in str(e.value)

    def test_no_fields_error(self, iris):
        algo_options = {'feature_variables': []}
        expected_error = 'You must specify one field'
        for is_acf in [True, False]:
            with raises(RuntimeError) as e:
                self.setup(iris, algo_options, acf=is_acf)
            assert expected_error in str(e.value)

    def test_pacf_categorical(self, iris):
        algo_options = {'feature_variables': ['species']}
        with raises(RuntimeError):
            self.setup(iris, algo_options, False)

    def test_acf_categorical(self, iris):
        algo_options = {'feature_variables': ['species']}
        with raises(RuntimeError):
            self.setup(iris, algo_options, True)

    def test_more_nulls_than_lags(self, iris):
        algo_options = {'feature_variables': ['petal_length'], 'params': {'k': '110'}}
        # Create mask for one species
        mask = iris['species'] != 'setosa'

        # Set those species to nan
        iris[mask] = np.nan

        # Expect a warning message regarding the number of null events dropped being passed to messages.warn as input
        with mock.patch('algos.ACF.messages.warn') as mock_warn:
            # should raise error since len(df) == 150, null == 100, non null == 50
            # but the k value == 110 which is greater than non null
            with raises(RuntimeError):
                self.setup(iris, algo_options, True)
            # Verify warning message
            mock_warn.assert_called_with(
                '{} events with nulls were dropped.'.format(np.sum(mask))
            )
