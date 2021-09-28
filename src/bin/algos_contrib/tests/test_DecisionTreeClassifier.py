#!/usr/bin/python

import json
import pytest
from algos.DecisionTreeClassifier import DecisionTreeClassifier


class TestDecisionTreeClassifier(object):
    def test_fit(self, iris):
        algo_options = {'target_variable': ['species'], 'feature_variables': ['petal_length']}
        DTC = DecisionTreeClassifier(algo_options)
        DTC.target_variable = algo_options['target_variable'][0]
        DTC.feature_variables = algo_options['feature_variables']
        DTC.fit(iris, algo_options)
        assert len(DTC.classes) == 3
        return DTC

    def test_invalid_criterion(self, iris):
        algo_options = {
            'target_variable': ['species'],
            'feature_variables': ['petal_length'],
            'params': {'criterion': 'not_gini'},
        }
        with pytest.raises(RuntimeError):
            DecisionTreeClassifier(algo_options)

    def test_invalid_splitter(self, iris):
        algo_options = {
            'target_variable': ['species'],
            'feature_variables': ['petal_length'],
            'params': {'splitter': 'not_random'},
        }
        with pytest.raises(RuntimeError):
            DecisionTreeClassifier(algo_options)

    def test_summary_json(self, iris):
        algo_options = {'target_variable': ['species'], 'feature_variables': ['petal_length']}
        DTC = DecisionTreeClassifier(algo_options)
        DTC.target_variable = algo_options['target_variable'][0]
        DTC.feature_variables = algo_options['feature_variables']
        DTC.fit(iris, algo_options)
        summary_options = {'params': {'json': 't', 'limit': '5'}}
        df = DTC.summary(options=summary_options)
        assert len(df) == 1
        json_string = df[df.columns[0]].values[0]
        json_summary = json.loads(json_string)
        correct_keys = set(['count', 'left child', 'impurity', 'split', 'right child', 'class'])
        assert set(json_summary.keys()) == correct_keys

    def test_summary_str(self, iris):
        algo_options = {'target_variable': ['species'], 'feature_variables': ['petal_length']}
        DTC = DecisionTreeClassifier(algo_options)
        DTC.target_variable = algo_options['target_variable'][0]
        DTC.feature_variables = algo_options['feature_variables']
        DTC.fit(iris, algo_options)
        summary_options = {'params': {'json': 'f', 'limit': '5'}}
        string = DTC.summary(options=summary_options)
        assert 'Decision Tree Summary' in string
