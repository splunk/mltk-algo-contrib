#!/usr/bin/env python

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

import cexc
from base import BaseAlgo
from util import df_util
from util.algo_util import alpha_to_confidence_interval, confidence_interval_to_alpha
from util.param_util import convert_params

messages = cexc.get_messages_logger()


class ACF(BaseAlgo):
    """Compute autocorrelation function."""

    def __init__(self, options):

        self.handle_options(options)

        params = options.get('params', {})
        converted_params = convert_params(
            params, ints=['k', 'conf_interval'], bools=['fft'], aliases={'k': 'nlags'}
        )

        # Set the default name to be used so that PACF can override
        self.default_name = 'acf({})'

        # Set the lags, alpha and fft parameters
        self.nlags = converted_params.pop('nlags', 40)
        self.fft = converted_params.pop('fft', False)

        conf_int = converted_params.pop('conf_interval', 95)
        if conf_int <= 0 or conf_int >= 100:
            raise RuntimeError('conf_interval cannot be less than 1 or more than 99.')
        if self.nlags <= 0:
            raise RuntimeError('k must be greater than 0.')
        self.alpha = confidence_interval_to_alpha(conf_int)

    @staticmethod
    def handle_options(options):
        """Ensure features are present but no target variable.

        Args:
            options (dict): algorithm options

        Raises:
            RuntimeError
        """
        feature_vars = options.get('feature_variables', [])
        target_vars = options.get('target_variable', [])

        if len(feature_vars) != 1:
            raise RuntimeError('You must specify one field.')

        if len(target_vars) != 0:
            raise RuntimeError('You cannot use from clause here.')

    def _calculate(self, df):
        """Calculate the ACF.

        Args:
            X (dataframe): input data

        Returns:
            autocors (array): array of autocorrelations
            conf_int (array): array of confidence intervals
        """
        autocors, conf_int = acf(x=df.values, nlags=self.nlags, alpha=self.alpha, fft=self.fft)
        return autocors, conf_int

    def fit(self, df, options):
        X = df.copy()

        X, nans, _ = df_util.prepare_features(
            X=X,
            variables=self.feature_variables,
            mlspl_limits=options.get('mlspl_limits'),
            get_dummies=False,
        )

        number_of_nulls = nans.sum()
        if number_of_nulls > 0:
            messages.warn('{} events with nulls were dropped.'.format(number_of_nulls))

        if self.nlags >= len(X):
            raise RuntimeError('k must be less than number of events.')

        # Only fields allowed (in case fields expanded through glob matching).
        if len(self.feature_variables) > 1:
            temp = 'You must specify only one field. Multiple fields found: {}'
            err = temp.format(', '.join(self.feature_variables))
            raise RuntimeError(err)

        # Only numeric inputs allowed.
        if X[self.feature_variables].dtypes.tolist()[0] == object:
            temp = '{} contains non-numeric data. {} only accepts numeric data.'
            err = temp.format(self.feature_variables[0], self.__class__.__name__)
            raise RuntimeError(err)

        # Get calculation
        autocors, conf_int = self._calculate(X)
        conf_int = conf_int - conf_int.mean(1)[:, None]

        # autocors[:, None] converts 1D-array to 2D for concatenation match
        autocors_2d = autocors[:, None]
        stacked = np.concatenate([autocors_2d, conf_int], axis=1)

        # Get the default name
        output_name = options.get('output_name', self.feature_variables[0])
        name = self.default_name.format(output_name)

        # Lower and upper names
        confidence_interval = alpha_to_confidence_interval(self.alpha)
        lower_name = 'lower{}({})'.format(confidence_interval, name)
        upper_name = 'upper{}({})'.format(confidence_interval, name)

        # Splunk arranges columns via ascii ordering
        # So the capital L on Lag ensures it will be in the leftmost column
        output_names = ['Lag', name, lower_name, upper_name]

        output = pd.DataFrame(stacked)
        output = output.reset_index()
        output.columns = output_names

        return output
