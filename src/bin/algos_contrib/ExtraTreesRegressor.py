#!/usr/bin/env python

from pandas import DataFrame
from sklearn.ensemble import ExtraTreesRegressor as _ExtraTreesRegressor

from base import RegressorMixin, BaseAlgo
from util.param_util import convert_params
from util.algo_util import handle_max_features
from codec import codecs_manager


class ExtraTreesRegressor(RegressorMixin, BaseAlgo):
    def __init__(self, options):
        self.handle_options(options)
        params = options.get('params', {})
        out_params = convert_params(
            params,
            floats=['max_samples', 'min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf', 'max_features', 'min_impurity_split'],
            bools=['bootstrap', 'oob_score', 'warm_start'],
            ints=['n_estimators', 'max_depth', 'max_leaf_nodes', 'min_impurity_decrease'],
            strs=['criterion'],
        )

        self.estimator = _ExtraTreesRegressor(**out_params)


    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec, TreeCodec

        codecs_manager.add_codec('algos.ExtraTreesRegressor', 'ExtraTreesRegressor', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.ensemble.forest', 'ExtraTreesRegressor', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.tree.tree', 'ExtraTreeRegressor', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.tree._tree', 'Tree', TreeCodec)
