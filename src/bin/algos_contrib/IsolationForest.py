#!/usr/bin/env python

from sklearn.ensemble import IsolationForest as _IsolationForest

from base import BaseAlgo, ClustererMixin
from codec import codecs_manager
from util.param_util import convert_params


class IsolationForest(ClustererMixin, BaseAlgo):

    def __init__(self, options):
        self.handle_options(options)

        out_params = convert_params(
            options.get('params', {}),
            floats=['contamination', 'max_features'],
            ints=['n_estimators', 'random_state'],
            bools=['bootstrap'],
            strs=['max_samples'],
        )

        self.estimator = _IsolationForest(**out_params)

    def rename_output(self, default_names, new_names):
        return new_names if new_names is not None else 'isNormal'


    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec, TreeCodec
        codecs_manager.add_codec('algos_contrib.IsolationForest', 'IsolationForest', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.ensemble.iforest', 'IsolationForest', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.tree.tree', 'ExtraTreeRegressor', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.tree._tree', 'Tree', TreeCodec)

