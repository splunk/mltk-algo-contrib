#!/usr/bin/python

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB as GaussianSK


from algos.GaussianNB import GaussianNB


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
        gaussNB = GaussianNB(algo_opt)
        gaussNB.target_variable = algo_opt['target_variable'][0]
        gaussNB.feature_variables = algo_opt['feature_variables']
        gaussNB.fit(data, algo_opt)
        return gaussNB

    def test_summary_values(self):
        """Test to verify actual values generated against predicted for feature probabilities"""
        gaussNB = self.test_init()
        summary_options = {'model_name': 'fake', 'mlspl_limits': {}}
        summary_df = gaussNB.summary(summary_options)
        feature_columns = [col for col in summary_df.columns if 'variance' in col]
        predicted = summary_df[feature_columns[0]]
        # calculate actual values from sckit-learn
        options = options_skeleton()
        gaussSK = GaussianSK()
        df1 = df.dropna()
        gaussSK.fit(df1[options['feature_variables']], df1[options['target_variable']])
        actual = gaussSK.sigma_[:, 0]
        assert np.isclose(predicted, actual, rtol=0.01).all()

        # Verify array size generated for classes
        assert len(summary_df['class_count']) == len(gaussSK.class_count_)

    def test_summary_columns(self):
        gaussNB = self.test_init()
        summary_options = {'model_name': 'fake', 'mlspl_limits': {}}
        summary_df = gaussNB.summary(summary_options)
        for class_names in ['class', 'class_count', 'class_prior']:
            assert class_names in summary_df.columns

        # Verify size of column arrays generated for features
        feature_columns = [col for col in summary_df.columns if 'variance' in col]
        assert len(feature_columns) > 0 and len(feature_columns) == len(
            gaussNB.feature_variables
        )

    def test_summary_depreciated(self):
        """Test to verify columns values and length for depreciated output"""
        algo_options = options_skeleton()

        # adding a dummy column for non-numeric features
        algo_options['feature_variables'] = algo_options['feature_variables'] + [
            'dummy_field_color'
        ]
        gaussNB = self.test_init(algo_options, df)
        summary_options = {'model_name': 'fake', 'mlspl_limits': {}}
        summary_df = gaussNB.summary(summary_options)
        assert set(summary_df.columns) == {'class', 'class_count', 'class_prior'}
