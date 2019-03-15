#! /usr/bin/env python


import pandas as pd
from sklearn.preprocessing import QuantileTransformer as _QuantileTransformer

from base import BaseAlgo, TransformerMixin
from codec import codecs_manager
from util.param_util import convert_params
from util import df_util


class QuantileTransformer(TransformerMixin, BaseAlgo):

    def __init__(self, options):
        self.handle_options(options)

        out_params = convert_params(
            options.get('params', {}),
            bools=['copy'],
            ints=['n_quantiles'],
            strs=['output_distribution']
        )
        self.estimator = _QuantileTransformer(**out_params)
        self.columns = None

    def rename_output(self, default_names, new_names=None):
        if new_names is None:
            new_names = 'QT'
        output_names = [new_names + '_' + feature for feature in self.columns]
        return output_names

    def summary(self, options):
        if len(options) != 2:  # only model name and mlspl_limits
            raise RuntimeError('"%s" models do not take options for summarization' % self.__class__.__name__)
        return pd.DataFrame({'fields': self.columns})

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec
        codecs_manager.add_codec('algos.QuantileTransformer', 'QuantileTransformer', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.preprocessing.data', 'QuantileTransformer', SimpleObjectCodec)
