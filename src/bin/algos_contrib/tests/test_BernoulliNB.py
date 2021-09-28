#!/usr/bin/python

import numpy as np
import pandas as pd

from algos.BernoulliNB import BernoulliNB
from sklearn.naive_bayes import BernoulliNB as BernoulliSK

df = pd.DataFrame(
    {
        'petal_length': [1, 5, 3, np.nan, 5.6, 4.2],
        'petal_width': [np.nan, 4, 2, 6.4, 2.6, 1.2],
        'dummy_field_color': ['green', 'green', 'blue', 'red', 'red', 'red'],
        'species': [
            'iris_setosa',
            'iris_setosa',
            'iris_versicolor',
            'iris_virginica',
            'iris_virginica',
            'iris_virginica',
        ],
    }
)


def options_skeleton(params=None):
    """Returns skeleton for options"""
    if params is None:
        params = {}
    return {
        'feature_variables': ['petal_length', 'petal_width'],
        'target_variable': ['species'],
        'params': params,
    }


class TestBernoulliNBClassifier(object):
    def test_init(self, algo_opt={}, data=df):
        if len(list(algo_opt.items())) == 0:
            algo_opt = options_skeleton()
        bernNB = BernoulliNB(algo_opt)
        bernNB.target_variable = algo_opt['target_variable'][0]
        bernNB.feature_variables = algo_opt['feature_variables']
        bernNB.fit(data, algo_opt)
        return bernNB

    def test_summary(self):
        bernNB = self.test_init()
        summary_options = {'model_name': 'fake', 'mlspl_limits': {}}
        summary_df = bernNB.summary(summary_options)

        assert 'class_count' in summary_df.columns
        # Verify size of array generated for features
        feature_columns = [col for col in summary_df.columns if 'log_prob' in col]
        assert not len(feature_columns) == 0
        # test cases for non-depreciated output
        assert len(feature_columns) == len(bernNB.feature_variables)

        # Verify actual values generated against predicted for feature log probability
        options = options_skeleton()
        bernSK = BernoulliSK()
        df1 = df.dropna()
        bernSK.fit(df1[options['feature_variables']], df1[options['target_variable']])
        predicted = summary_df[feature_columns[0]]
        actual = bernSK.feature_log_prob_[:, 0]
        assert np.isclose(predicted, actual, rtol=0.01).all()

        # Verify size of array generated for classes
        assert len(summary_df['class_count']) == len(bernSK.class_count_)

    # Tests columns generated for depreciated output
    def test_summary_depreciated(self):
        algo_options = options_skeleton()

        # adding a dummy column for non-numeric features
        algo_options['feature_variables'] = algo_options['feature_variables'] + [
            'dummy_field_color'
        ]
        bernNB = self.test_init(algo_options, df)

        summary_options = {'model_name': 'fake', 'mlspl_limits': {}}
        summary_df = bernNB.summary(summary_options)
        # verify for length of class information scores : ['class','class_count','class_log_prior']
        assert len(summary_df.columns) == 3
        assert 'class_log_prior' in summary_df.columns
