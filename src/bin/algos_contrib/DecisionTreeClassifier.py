#!/usr/bin/env python

from sklearn.tree import DecisionTreeClassifier as _DecisionTreeClassifier

from base import ClassifierMixin, BaseAlgo
from codec import codecs_manager
from util.param_util import convert_params
from util.algo_util import tree_summary


class DecisionTreeClassifier(ClassifierMixin, BaseAlgo):
    def __init__(self, options):
        self.handle_options(options)

        out_params = convert_params(
            options.get('params', {}),
            ints=['random_state', 'max_depth', 'min_samples_split', 'max_leaf_nodes'],
            strs=['criterion', 'splitter', 'max_features'],
        )

        # whitelist valid values for criterion, as error raised by sklearn for invalid values is uninformative
        if 'criterion' in out_params:
            try:
                assert out_params['criterion'] in ['gini', 'entropy']
            except AssertionError:
                raise RuntimeError(
                    'Invalid value for option criterion: "%s"' % out_params['criterion']
                )

        # whitelist valid values for splitter, as error raised by sklearn for invalid values is uninformative
        if 'splitter' in out_params:
            try:
                assert out_params['splitter'] in ['best', 'random']
            except AssertionError:
                raise RuntimeError(
                    'Invalid value for option splitter: "%s"' % out_params['splitter']
                )

        if 'max_depth' not in out_params:
            out_params.setdefault('max_leaf_nodes', 2000)

        # EAFP... convert max_features to int or float if it is a number.
        try:
            out_params['max_features'] = float(out_params['max_features'])
            max_features_int = int(out_params['max_features'])
            if out_params['max_features'] == max_features_int:
                out_params['max_features'] = max_features_int
        except:
            pass

        self.estimator = _DecisionTreeClassifier(class_weight='balanced', **out_params)

    def summary(self, options):
        if 'args' in options:
            raise RuntimeError('Summarization does not take values other than parameters')
        return tree_summary(self, options)

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec, TreeCodec

        codecs_manager.add_codec(
            'algos.DecisionTreeClassifier', 'DecisionTreeClassifier', SimpleObjectCodec
        )
        codecs_manager.add_codec(
            'sklearn.tree._classes', 'DecisionTreeClassifier', SimpleObjectCodec
        )
        codecs_manager.add_codec('sklearn.tree._tree', 'Tree', TreeCodec)
