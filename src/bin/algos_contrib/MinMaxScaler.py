#!/usr/bin/env python

import pandas as pd
from sklearn.preprocessing import MinMaxScaler as _MinMaxScaler

from base import BaseAlgo, TransformerMixin
from codec import codecs_manager
from util.param_util import convert_params
from util import df_util


class MinMaxScaler(TransformerMixin, BaseAlgo):

    def __init__(self, options):
        self.handle_options(options)

        out_params = convert_params(
            options.get('params', {}),
            bools=['copy'],
            strs=['feature_range']
        )
        self.estimator = _MinMaxScaler(**out_params)
        self.columns = None

    def rename_output(self, default_names, new_names=None):
        if new_names is None:
            new_names = 'MMS'
        output_names = [new_names + '_' + feature for feature in self.columns]
        return output_names

    def partial_fit(self, df, options):
        # Make a copy of data, to not alter original dataframe
        X = df.copy()

        X, _, columns = df_util.prepare_features(
            X=X,
            variables=self.feature_variables,
            mlspl_limits=options.get('mlspl_limits'),
        )
        if self.columns is not None:
            df_util.handle_new_categorical_values(X, None, options, self.columns)
            if X.empty:
                return
        else:
            self.columns = columns
        self.estimator.partial_fit(X)

    def summary(self, options):
        if len(options) != 2:  # only model name and mlspl_limits
            raise RuntimeError('"%s" models do not take options for summarization' % self.__class__.__name__)
        return pd.DataFrame({'fields': self.columns,
                             'mean': self.estimator.mean_,
                             'var': self.estimator.var_,
                             'scale': self.estimator.scale_})

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec
        codecs_manager.add_codec('algos_contrib.MinMaxScaler', 'MinMaxScaler', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.preprocessing.data', 'MinMaxScaler', SimpleObjectCodec)
