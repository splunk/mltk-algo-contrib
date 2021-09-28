#!/usr/bin/env python

import copy
import numpy as np
import pandas as pd

from sklearn.feature_selection import GenericUnivariateSelect, f_classif, f_regression
from sklearn import __version__ as sklearn_version

import cexc
from distutils.version import StrictVersion
from base import BaseAlgo
from codec import codecs_manager
from codec.codecs import BaseCodec
from util.param_util import convert_params
from util import df_util


messages = cexc.get_messages_logger()

required_version = '0.19.0'


def has_required_version():
    return StrictVersion(sklearn_version) >= StrictVersion(required_version)


class GenericUnivariateSelectCodec(BaseCodec):
    @classmethod
    def encode(cls, obj):
        obj = copy.deepcopy(obj)
        obj.score_func = obj.score_func.__name__

        return {
            '__mlspl_type': [type(obj).__module__, type(obj).__name__],
            'dict': obj.__dict__,
        }

    @classmethod
    def decode(cls, obj):
        from sklearn.feature_selection import f_classif, f_regression, GenericUnivariateSelect

        new_obj = GenericUnivariateSelect.__new__(GenericUnivariateSelect)
        new_obj.__dict__ = obj['dict']

        if new_obj.score_func == 'f_classif':
            new_obj.score_func = f_classif
        elif new_obj.score_func == 'f_regression':
            new_obj.score_func = f_regression
        else:
            raise ValueError(
                'Unsupported GenericUnivariateSelect.score_func "%s"' % new_obj.score_func
            )

        return new_obj


class FieldSelector(BaseAlgo):
    def __init__(self, options):
        self.handle_options(options)

        out_params = convert_params(
            options.get('params', {}),
            floats=['param'],
            strs=['type', 'mode'],
            aliases={'type': 'score_func'},
        )

        if 'score_func' not in out_params:
            out_params['score_func'] = f_classif
        else:
            if out_params['score_func'].lower() == 'categorical':
                out_params['score_func'] = f_classif
            elif out_params['score_func'].lower() in ['numerical', 'numeric']:
                out_params['score_func'] = f_regression
            else:
                raise RuntimeError('type can either be categorical or numeric.')

        if 'mode' in out_params:
            if out_params['mode'] not in ('k_best', 'fpr', 'fdr', 'fwe', 'percentile'):
                raise RuntimeError(
                    'mode can only be one of the following: fdr, fpr, fwe, k_best, and percentile'
                )

            if out_params['mode'] in ['fpr', 'fdr', 'fwe']:
                if 'param' in out_params:
                    if not 0 < out_params['param'] < 1:
                        msg = 'Invalid param value for mode {}: param must be between 0 and 1.'.format(
                            out_params['mode']
                        )
                        raise ValueError(msg)

        # k_best and percentile require integer param
        if 'param' in out_params and out_params.get('mode') not in ['fdr', 'fpr', 'fwe']:
            original_value = out_params['param']
            out_params['param'] = int(out_params['param'])
            if out_params['param'] != original_value:
                msg = 'param value {} is not an integer; mode={} requires an integer.'
                msg = msg.format(original_value, out_params.get('mode', 'percentile'))
                raise ValueError(msg)

        self.estimator = GenericUnivariateSelect(**out_params)

    def handle_options(self, options):
        if (
            len(options.get('target_variable', [])) != 1
            or len(options.get('feature_variables', [])) == 0
        ):
            raise RuntimeError('Syntax error: expected "<target> FROM <field> ..."')

    def fit(self, df, options):
        # Make a copy of data, to not alter original dataframe
        X = df.copy()

        relevant_variables = self.feature_variables + [self.target_variable]
        X, y, self.columns = df_util.prepare_features_and_target(
            X=X,
            variables=relevant_variables,
            target=self.target_variable,
            mlspl_limits=options.get('mlspl_limits'),
        )
        self.estimator.fit(X.values, y.values)

    def apply(self, df, options):
        # Make a copy of data, to not alter original dataframe
        X = df.copy()

        X, nans, columns = df_util.prepare_features(
            X=X,
            variables=self.feature_variables,
            final_columns=self.columns,
            mlspl_limits=options.get('mlspl_limits'),
        )
        y_hat = self.estimator.transform(X.values)
        mask = self.estimator.get_support()
        columns_select = np.array(self.columns)[mask]
        width = len(columns_select)

        if width == 0:
            messages.warn(
                'No fields pass the current configuration. Consider changing your parameters.'
            )

        default_name = 'fs'
        output_name = options.get('output_name', default_name)
        output_names = [output_name + '_%s' % x for x in columns_select]

        output = df_util.create_output_dataframe(
            y_hat=y_hat, nans=nans, output_names=output_names
        )

        df = df_util.merge_predictions(df, output)
        return df

    def summary(self, options):
        """Only model_name and mlspl_limits are supported for summary"""
        if len(options) != 2:
            raise RuntimeError(
                '{class_name} models do not take options for summarization'.format(
                    class_name=self.__class__.__name__
                )
            )

        df = pd.DataFrame(
            {
                'feature_variables': self.feature_variables,
                'score': self.estimator.scores_.round(4),
                'p-value': self.estimator.pvalues_.round(4),
            },
            index=self.feature_variables,
        )
        return df

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec

        codecs_manager.add_codec('algos.FieldSelector', 'FieldSelector', SimpleObjectCodec)
        codecs_manager.add_codec(
            'sklearn.feature_selection._univariate_selection',
            'GenericUnivariateSelect',
            GenericUnivariateSelectCodec,
        )
