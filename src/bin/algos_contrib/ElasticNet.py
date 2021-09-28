#!/usr/bin/env python

import pandas as pd
from sklearn.linear_model import ElasticNet as _ElasticNet

from base import RegressorMixin, BaseAlgo
from codec import codecs_manager
from util.param_util import convert_params


class ElasticNet(RegressorMixin, BaseAlgo):
    def __init__(self, options):
        self.handle_options(options)

        out_params = convert_params(
            options.get('params', {}),
            bools=['fit_intercept', 'normalize'],
            floats=['alpha', 'l1_ratio'],
        )

        if 'l1_ratio' in out_params:
            if out_params['l1_ratio'] < 0 or out_params['l1_ratio'] > 1:
                raise RuntimeError('l1_ratio must be >= 0 and <= 1')

        self.estimator = _ElasticNet(**out_params)

    def summary(self, options):
        if len(options) != 2:  # only model name and mlspl_limits
            raise RuntimeError(
                '"%s" models do not take options for summarization' % self.__class__.__name__
            )

        df = pd.DataFrame(
            {'feature': self.columns, 'coefficient': self.estimator.coef_.ravel()}
        )
        idf = pd.DataFrame(
            {'feature': ['_intercept'], 'coefficient': [self.estimator.intercept_]}
        )
        return pd.concat([df, idf])

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec

        codecs_manager.add_codec('algos.ElasticNet', 'ElasticNet', SimpleObjectCodec)
        codecs_manager.add_codec(
            'sklearn.linear_model._coordinate_descent', 'ElasticNet', SimpleObjectCodec
        )
