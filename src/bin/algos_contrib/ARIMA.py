#!/usr/bin/env python

import datetime

import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA as _ARIMA
from statsmodels.tools.sm_exceptions import MissingDataError

import cexc
from base import BaseAlgo
from util.algo_util import confidence_interval_to_alpha
from util.algo_util import alpha_to_confidence_interval
from util.param_util import convert_params
from util import df_util


class ARIMA(BaseAlgo):
    def __init__(self, options):
        self.handle_options(options)

        params = convert_params(
            options.get('params', {}),
            strs=['order'],
            ints=['forecast_k', 'conf_interval', 'holdback'],
            aliases={'forecast_k': 'steps'},
        )
        self.out_params = dict(model_params=dict(), forecast_function_params=dict())

        if 'order' in params:
            # statsmodels wants a tuple for order of the model for the number of AR parameters,
            # differences, and MA parameters.
            # SPL won't accept a tuple as an option's value, so the next few lines will make it possible for the
            # user to configure order.
            try:
                self.out_params['model_params']['order'] = tuple(
                    int(i) for i in params['order'].split('-')
                )
                assert len(self.out_params['model_params']['order']) == 3
            except:
                raise RuntimeError(
                    'Syntax Error: order requires three non-negative integer values, e.g. order=4-1-2'
                )
        else:
            raise RuntimeError(
                'Order of model is missing. It is required for fitting. e.g. order=<No. of AR>-'
                '<Parameters-No. of Differences>-<No. of MA Parameters>'
            )

        # Default steps set to zero
        steps = params.get('steps', 0)
        self._test_forecast_k(steps)
        self.out_params['forecast_function_params']['steps'] = steps

        if 'conf_interval' in params:
            self.out_params['forecast_function_params']['alpha'] = confidence_interval_to_alpha(
                params['conf_interval']
            )
        else:
            self.out_params['forecast_function_params'][
                'alpha'
            ] = 0.05  # the default value that ARIMAResults.forecast uses.

        if 'holdback' in params:
            self._test_holdback(params['holdback'])
            self.holdback = params.pop('holdback')
            # The required ratio of invariant time frequencies (deltas)
            # Between rows
            self.freq_threshold = 1.0
        else:
            self.holdback = 0
            self.freq_threshold = 0.9

        # Dealing with Missing data
        # if 'missing' in params and params['missing'] in ['raise', 'drop']:
        #     self.out_params['model_params']['missing'] = params['missing']
        # else:
        self.out_params['model_params']['missing'] = 'raise'

    def handle_options(self, options):
        """Utility to ensure there is a feature_variables and or _time"""
        self.feature_variables = options.get('feature_variables', [])
        number_of_vars = len(self.feature_variables)
        if number_of_vars == 0 or number_of_vars > 2:
            raise RuntimeError('Syntax error: expected " _time, <field>."')

        if len(self.feature_variables) == 2:
            if '_time' in self.feature_variables:
                self.time_series = (set(self.feature_variables) - set(['_time'])).pop()
            else:
                raise RuntimeError('Syntax error: if two fields given, one should be _time.')
        elif len(self.feature_variables) == 1:
            self.time_series = self.feature_variables[0]
        self.used_variables = self.feature_variables

    @staticmethod
    def _test_forecast_k(x):
        if x < 0:
            raise RuntimeError('forecast_k should be an integer equal to or greater than 0.')

    @staticmethod
    def _test_holdback(hd):
        if hd < 0:
            raise RuntimeError('holdback should be an integer equal to or greater than 0.')

    @staticmethod
    def _find_freq(X, threshold):
        """
        Calculates the dominant value of differences between two consequent timestamps.
        Checks if its frequency is above the threshold.
        """

        err = (
            'Sampling is irregular. ARIMA will use the median '
            'time interval: {} '
            'Try removing \'_time\' from the \'fit\''
        )

        y = np.diff(X)
        median = np.median(y)
        ratio = np.mean(y == median)

        timestr = "{:0>8}".format(str(datetime.timedelta(seconds=median)))
        if ratio < threshold:
            cexc.messages.warn(err.format(timestr))

        return median

    @staticmethod
    def _generate_timestamps_for_forecast(forecast_k, freq, last_timestamp):
        return np.arange(1, forecast_k + 1) * freq + last_timestamp

    def _fit(self, X):
        for variable in self.feature_variables:
            df_util.assert_field_present(X, variable)
        df_util.drop_unused_fields(X, self.feature_variables)
        df_util.assert_any_fields(X)
        df_util.assert_any_rows(X)

        if X[self.time_series].dtype == object:
            raise ValueError(
                '%s contains non-numeric data. ARIMA only accepts numeric data.'
                % self.time_series
            )
        X[self.time_series] = X[self.time_series].astype(float)

        try:
            self.estimator = _ARIMA(
                X[self.time_series].values,
                order=self.out_params['model_params']['order'],
                missing=self.out_params['model_params']['missing'],
            ).fit(disp=False)
        except ValueError as e:
            if 'stationary' in str(e):
                raise ValueError(
                    "The computed initial AR coefficients are not "
                    "stationary. You should induce stationarity by choosing a different model order."
                )
            elif 'invertible' in str(e):
                raise ValueError(
                    "The computed initial MA coefficients are not invertible. "
                    "You should induce invertibility by choosing a different model order."
                )
            else:
                cexc.log_traceback()
                raise ValueError(e)
        except MissingDataError:
            raise RuntimeError(
                'Empty or null values are not supported in %s. '
                'If using timechart, try using a larger span.' % self.time_series
            )
        except Exception as e:
            cexc.log_traceback()
            raise RuntimeError(e)

        # Saving the _time but not as a part of the ARIMA structure but as new attribute for ARIMA.
        if '_time' in self.feature_variables:
            freq = self._find_freq(X['_time'].values, self.freq_threshold)
            self.estimator.datetime_information = dict(
                ver=0,
                _time=X['_time'].values,
                freq=freq,
                # in seconds (unix epoch)
                first_timestamp=X['_time'].values[0],
                last_timestamp=X['_time'].values[-1],
                length=len(X),
            )
        else:
            self.estimator.datetime_information = dict(
                ver=0, _time=None, freq=None, first_time=None, last_time=None, length=len(X)
            )

    def _forecast(self, options, output_name=None):
        forecast_output = self.estimator.forecast(**options)

        lower_name = 'lower%d(%s)' % (
            alpha_to_confidence_interval(options['alpha']),
            output_name,
        )
        upper_name = 'upper%d(%s)' % (
            alpha_to_confidence_interval(options['alpha']),
            output_name,
        )
        # std_err = 'standard_error'

        output = pd.DataFrame(
            columns=[self.time_series, output_name, lower_name, upper_name],
            index=list(
                range(
                    self.estimator.datetime_information['length'],
                    self.estimator.datetime_information['length'] + options['steps'],
                )
            ),
        )
        output[output_name] = forecast_output[0]
        # output[std_err] = forecast_output[1]
        output[lower_name] = forecast_output[2][:, 0]
        output[upper_name] = forecast_output[2][:, 1]
        if (
            self.estimator.datetime_information['ver'] == 0
            and self.estimator.datetime_information['freq'] is not None
        ):
            output['_time'] = self._generate_timestamps_for_forecast(
                options['steps'],
                self.estimator.datetime_information['freq'],
                self.estimator.datetime_information['last_timestamp'],
            )

        return output

    def fit(self, df, options=None):
        # Make a copy of data, to not alter original dataframe
        X = df.copy()

        start = self.out_params['model_params']['order'][1]

        if self.holdback > 0:
            if self.holdback >= len(df):
                raise RuntimeError(
                    'holdback value ({}) must be less than the number of events ({})'.format(
                        self.holdback, len(df)
                    )
                )
            X = X.iloc[: -self.holdback]

        self._fit(X)

        default_name = 'predicted(%s)' % self.time_series
        output_name = options.get('output_name', default_name)

        df[output_name] = np.nan

        if self.out_params['model_params']['order'][1] > 0:  # d > 0
            predicted_vals = self.estimator.predict(typ='levels')
        else:
            # d=0. When d=0, ARIMA switches to ARMA under the hood and ARMA.predict() doesn't have the 'typ'
            # parameter, so we can't set typ='levels' as above.
            predicted_vals = self.estimator.predict()
        df[output_name].iloc[start : len(df) - self.holdback] = predicted_vals

        if self.out_params['forecast_function_params']['steps'] > 0:
            params = self.out_params['forecast_function_params']
            forecast_output = self._forecast(params, output_name)
            extra_columns = set(forecast_output.columns).difference(df)
            for col in extra_columns:
                df[col] = np.nan

            df = df.combine_first(forecast_output)

        df = df.sort_index(ascending=True)
        return df
