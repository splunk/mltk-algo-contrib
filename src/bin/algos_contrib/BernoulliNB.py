#!/usr/bin/env python

from sklearn.naive_bayes import BernoulliNB as _BernoulliNB

from base import BaseAlgo, ClassifierMixin
from util.param_util import convert_params
from codec import codecs_manager
from util import df_util
import pandas as pd
import cexc


class BernoulliNB(ClassifierMixin, BaseAlgo):
    def __init__(self, options):
        self.handle_options(options)

        out_params = convert_params(
            options.get('params', {}), floats=['alpha', 'binarize'], bools=['fit_prior']
        )

        self.estimator = _BernoulliNB(**out_params)

    def summary(self, options):
        """Only model_name and mlspl_limits are supported for summary"""
        # Can be modified further to include optional parameters in summary like feature_count = true/false
        if len(options) != 2:
            msg = '"%s" models do not take options for summarization' % self.__class__.__name__
            raise RuntimeError(msg)

        # DataFrame to include class scores
        df = pd.DataFrame(
            {
                'class': self.estimator.classes_,
                'class_count': self.estimator.class_count_.astype(int),
                'class_log_prior': self.estimator.class_log_prior_.round(3),
            },
            index=self.estimator.classes_,
        )

        # renaming column names for display of feature_log_probability
        feature_variables = [
            'log_prob({})'.format(feature) for feature in self.feature_variables
        ]

        feature_log_arr = self.estimator.feature_log_prob_.round(3)

        # The default behaviour when encountering categorical fields is to one-hot-encode,
        # causing the number of feature columns to grow by the number of unique labels in the field.
        # currently we do not support feature-summary information for such cases.
        if len(feature_variables) == feature_log_arr.shape[1]:

            # Create dataFrame to include feature probability scores per class
            df_feature_score = pd.DataFrame(
                columns=feature_variables, data=feature_log_arr, index=self.estimator.classes_
            )
            df = df_util.merge_predictions(df, df_feature_score)
        else:
            # Raise a warning to inform the user about depreciated columns, with basic class scores.
            cexc.messages.warn(
                "Unable to display log-probability information for features when categorical fields are supplied."
            )

        return df

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec

        codecs_manager.add_codec('algos.BernoulliNB', 'BernoulliNB', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.naive_bayes', 'BernoulliNB', SimpleObjectCodec)
