#!/usr/bin/env python

import pandas as pd
from sklearn.naive_bayes import GaussianNB as _GaussianNB


import cexc
from base import BaseAlgo, ClassifierMixin
from codec import codecs_manager
from util import df_util


class GaussianNB(ClassifierMixin, BaseAlgo):
    def __init__(self, options):
        self.handle_options(options)
        self.estimator = _GaussianNB()

    def summary(self, options):
        """Only model_name and mlspl_limits are supported for summary"""
        if len(options) != 2:
            msg = '"%s" models do not take options for summarization' % self.__class__.__name__
            raise RuntimeError(msg)

        classes = self.estimator.classes_
        # DataFrame to include class scores
        df = pd.DataFrame(
            {
                'class': classes,
                'class_count': self.estimator.class_count_.astype(int),
                'class_prior': self.estimator.class_prior_.round(3),
            },
            index=classes,
        )
        # renaming column names for display of feature_log_probability
        feature_scores = ['variance({})'.format(feature) for feature in self.feature_variables]
        feature_variance = self.estimator.sigma_.round(3)

        # The default behaviour when encountering categorical fields is to one-hot-encode,
        # causing the number of feature columns to grow by the number of unique labels in the field.
        # currently we do not support feature-summary information for such cases.
        if len(feature_scores) == feature_variance.shape[1]:

            # Create dataFrame to include feature probability scores per class
            df_feature_score = pd.DataFrame(
                columns=feature_scores, data=feature_variance, index=classes
            )
            df = df_util.merge_predictions(df, df_feature_score)
        else:
            # Raise a warning to inform the user about depreciated columns, with basic class scores.
            cexc.messages.warn(
                "Unable to display variance information for features when categorical fields are supplied."
            )
        return df

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec

        codecs_manager.add_codec('algos.GaussianNB', 'GaussianNB', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.naive_bayes', 'GaussianNB', SimpleObjectCodec)
