#!/usr/bin/env python

import numpy as np
from pytest import raises
from algos.ARIMA import ARIMA

import random

random.seed(42)
np.random.seed(42)


class TestARIMA(object):
    def test_fit(self, ts_voltage):
        options = {'feature_variables': ['Voltage'], 'params': {'order': '0-0-1'}}
        arima = ARIMA(options)
        df = arima.fit(ts_voltage, options)
        assert len(df) == 999
        assert len(df.columns) == 3
        the_sum = round(df['predicted(Voltage)'].sum())
        assert the_sum == 239800.0

    def test_fit_with_time(self, ts_voltage):
        options = {'feature_variables': ['Voltage', '_time'], 'params': {'order': '0-0-1'}}
        arima = ARIMA(options)
        df = arima.fit(ts_voltage, options)
        assert len(df) == 999
        assert len(df.columns) == 3
        the_sum = round(df['predicted(Voltage)'].sum())
        assert the_sum == 239800.0

    def test_forecast(self, ts_voltage):
        options = {
            'feature_variables': ['Voltage', '_time'],
            'params': {'order': '0-0-1', 'forecast_k': '100'},
        }
        arima = ARIMA(options)
        df = arima.fit(ts_voltage, options)
        assert len(df) == 1099
        assert len(df.columns) == 5
        the_sum = round(df['predicted(Voltage)'].sum())
        assert the_sum == 263802.0
        upper_lower = df[['lower95(predicted(Voltage))', 'upper95(predicted(Voltage))']]
        # There should only be confidence scores for the forecast_k amount
        assert set(upper_lower.notnull().sum().values) == set([100])

    def test_holdback(self, ts_voltage):
        clean = ts_voltage.copy()
        options = {
            'feature_variables': ['Voltage', '_time'],
            'params': {'order': '0-0-1', 'holdback': '100'},
        }
        arima = ARIMA(options)
        df = arima.fit(ts_voltage, options)
        preds = df['predicted(Voltage)']
        assert preds.iloc[-100:].isnull().sum() == 100
        no_preds = df.drop('predicted(Voltage)', axis=1)
        assert no_preds.shape == clean.shape
        assert set(no_preds.columns) == set(clean.columns)

    def test_holdback_too_big(self, ts_voltage):
        ts_voltage_small = ts_voltage.iloc[:50]
        options = {
            'feature_variables': ['Voltage', '_time'],
            'params': {'order': '0-0-1', 'holdback': '100'},
        }
        arima = ARIMA(options)
        with raises(RuntimeError):
            arima.fit(ts_voltage_small, options)

    def test_holdback_too_small(self, ts_voltage):
        options = {
            'feature_variables': ['Voltage', '_time'],
            'params': {'order': '0-0-1', 'holdback': '-1'},
        }
        with raises(RuntimeError) as e:
            ARIMA(options)
        assert 'holdback should be an integer equal to or greater than 0' in str(e.value)

    def test_holdback_forecast(self, ts_voltage):
        clean = ts_voltage.copy()
        options = {
            'feature_variables': ['Voltage', '_time'],
            'params': {'order': '0-0-1', 'holdback': '100', 'forecast_k': '100'},
        }
        arima = ARIMA(options)
        df = arima.fit(ts_voltage, options)
        upper_lower = df[['lower95(predicted(Voltage))', 'upper95(predicted(Voltage))']]
        # There should only be confidence scores for the forecast_k amount
        assert set(upper_lower.notnull().sum().values) == set([100])
        assert df.shape == (999, 5)

        # these are the predictions using the holdback param
        hold_back_preds = df[
            ['predicted(Voltage)', 'lower95(predicted(Voltage))', 'upper95(predicted(Voltage))']
        ]

        options = {
            'feature_variables': ['Voltage', '_time'],
            'params': {'order': '0-0-1', 'forecast_k': '100'},
        }

        manual_holdback = clean.iloc[:-100]
        arima = ARIMA(options)
        df = arima.fit(manual_holdback, options)

        # these are the predictions obtained by manually doing a holdback
        manual_holdback_preds = df[
            ['predicted(Voltage)', 'lower95(predicted(Voltage))', 'upper95(predicted(Voltage))']
        ]

        hbp = hold_back_preds[hold_back_preds.notnull()].values
        mhp = manual_holdback_preds[manual_holdback_preds.notnull()].values
        np.testing.assert_array_equal(hbp, mhp)

    @staticmethod
    def add_missing_time(df, frac):
        sample = df.sample(random_state=42, frac=frac).index.values
        time_loc = df.columns.get_loc('_time')
        df.iloc[sample, time_loc] = np.nan
        return df

    def test_missing_time_under_threshold(self, ts_voltage):
        # threshold = 90% must have the same delta between rows for _time
        ts_voltage = self.add_missing_time(ts_voltage, 0.09)
        ts_voltage.dropna(inplace=True)
        print((ts_voltage.info()))
        options = {
            'feature_variables': ['Voltage', '_time'],
            'params': {'order': '0-0-1', 'forecast_k': '100'},
        }
        arima = ARIMA(options)
        arima.fit(ts_voltage, options)

    def test_missing_feature_variable_field(self):
        options = {'feature_variables': [], 'params': {'order': '0-0-1', 'forecast_k': '100'}}
        with raises(RuntimeError) as e:
            ARIMA(options)
        assert 'expected " _time, <field>."' in str(e.value)

    def test_predict_with_differencing(self, ts_voltage):
        '''
        This test arises from MLA-1876.
        The bug reported in MLA-1876 is: when d > 0, the predicted
        values for the past were wrong and mostly nearly 0.
        So to test, we want the sum of all predicted
        values, past and future, to be relatively large.
        In this test, we predict 20 future values. Since 'Voltage' values
        are near 250 (but not bigger), we expect the sum of all predicted values
        to exceed 20*250. Previously, the bug caused this sum to be less
        than 20*250, because the past values were all close to 0.
        '''
        options = {
            'feature_variables': ['Voltage', '_time'],
            'params': {'order': '2-1-1', 'forecast_k': '20'},
        }
        arima = ARIMA(options)
        df = arima.fit(ts_voltage, options)
        the_sum = round(df['predicted(Voltage)'].sum())
        assert the_sum > 20 * 250
