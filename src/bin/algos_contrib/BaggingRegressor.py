#!/usr/bin/env python

from pandas import DataFrame
from sklearn.ensemble import BaggingRegressor as _BaggingRegressor

from base import RegressorMixin, BaseAlgo
from util.param_util import convert_params
from util.algo_util import handle_max_features
from codec import codecs_manager


class BaggingRegressor(RegressorMixin, BaseAlgo):
    def __init__(self, options):
        self.handle_options(options)
        params = options.get('params', {})
        out_params = convert_params(
            params,
            floats=['max_samples', 'max_features'],
            bools=['bootstrap', 'bootstrap_features', 'oob_score', 'warm_start'],
            ints=['n_estimators'],
        )

        self.estimator = _BaggingRegressor(**out_params)
        

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec, TreeCodec

        codecs_manager.add_codec('algos.BaggingRegressor', 'BaggingRegressor', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.ensemble.classes', 'BaggingRegressor', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.tree.tree', 'DecisionTreeRegressor', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.ensemble.weight_boosting', 'BaggingRegressor', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.ensemble.bagging', 'BaggingRegressor', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.tree._tree', 'Tree', TreeCodec)
