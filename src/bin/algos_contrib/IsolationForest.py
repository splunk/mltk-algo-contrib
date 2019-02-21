#!/usr/bin/env python

from sklearn.ensemble import IsolationForest as _IsolationForest
import numpy as np
import pandas as pd

from base import ClustererMixin, BaseAlgo
from codec import codecs_manager
from codec.codecs import BaseCodec
from codec.flatten import flatten, expand
from util import df_util
from util.param_util import convert_params
from cexc import get_messages_logger,get_logger

class IsolationForest(ClustererMixin, BaseAlgo):
    """
    This is the implementation wrapper around Isolation Forest from scikit-learn. It inherits methods from ClustererMixin and BaseAlgo.
    """
    def __init__(self,options):
        self.handle_options(options)
        out_params = convert_params(
            options.get('params',{}),
            ints = ['n_estimators','n_jobs','random_state','verbose'],
            floats = ['max_samples','contamination','max_features'],
            bools = ['bootstrap']
            )
        self.return_scores = out_params.pop('anomaly_score', True)

        # whitelist n_estimators > 0
        if 'n_estimators' in out_params and out_params['n_estimators']<=0:
            msg = 'Invalid value error: n_estimators must be greater than 0 and an integer, but found n_estimators="{}".'
            raise RuntimeError(msg.format(out_params['n_estimators']))
        
        # whitelist max_samples > 0 and < 1
        if 'max_samples' in out_params and out_params['max_samples']<0 and out_params['max_samples']>1:
            msg = 'Invalid value error: max_samples must be greater than 0 and a float, but found max_samples="{}".'
            raise RuntimeError(msg.format(out_params['max_samples']))
        
        #   whitelist contamination should be in (0.0, 0.5] as error raised by sklearn for values out of range
        if 'contamination' in out_params and not (0.0 < out_params['contamination'] <= 0.5):
            msg = (
                'Invalid value error: Valid values for contamination are in (0.0, 0.5], '
                'but found contamination="{}".'
            )
            raise RuntimeError(msg.format(out_params['contamination']))

        # whitelist max_features > 0 and < 1
        if 'max_features' in out_params and out_params['max_features']<0 and out_params['max_features']>1:
            msg = 'Invalid value error: max_features must be greater than 0, but found max_features="{}".'
            raise RuntimeError(msg.format(out_params['max_features']))

        
        self.estimator = _IsolationForest(**out_params)    


    def apply(self, df, options):
        # Make a copy of data, to not alter original dataframe
        logger = get_logger('IsolationForest Logger')
        X = df.copy()

        X, nans, _ = df_util.prepare_features(
            X=X,
            variables=self.feature_variables,
            final_columns=self.columns,
            mlspl_limits=options.get('mlspl_limits'),
        )

        # Multiplying the result by -1 to represent Outliers with 1 and Inliers/Normal points with 1.
        y_hat = self.estimator.predict(X.values)*-1
        # Printing the accuracy for prediction of outliers
        accuracy = "Accuracy: {}".format(str(round((list(y_hat).count(-1)*100)/y_hat.shape[0], 2)))
        logger.debug(accuracy)
        
        y_hat = y_hat.astype('str')

        #Assign output_name
        default_name = 'isOutlier'
        new_name = options.get('output_name', None)
        output_name = self.rename_output(default_names=default_name, new_names=new_name)

        # Create output dataframe
        output = df_util.create_output_dataframe(
            y_hat=y_hat, nans=nans, output_names=output_name
        )
        # Merge with original dataframe
        output = df_util.merge_predictions(df, output)
        return output

    def rename_output(self, default_names, new_names=None):
        """Utility hook to rename output.

        The default behavior is to take the default_names passed in and simply
        return them. If however a particular algo needs to rename the columns of
        the output, this method can be overridden.
        """
        return new_names if new_names is not None else default_names    


    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec, TreeCodec
        codecs_manager.add_codec('algos.IsolationForest', 'IsolationForest', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.ensemble.iforest', 'IsolationForest', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.tree.tree','ExtraTreeRegressor', ExtraTreeRegressorCodec)
        codecs_manager.add_codec('sklearn.tree._tree', 'Tree', TreeCodec)


class ExtraTreeRegressorCodec(BaseCodec):
    """
    This is an ExtraTreeRegressor Codec for saving the Isolation Forest base estimator to memory/file.
    """
    @classmethod
    def encode(cls, obj):
        import sklearn.tree
        assert type(obj) == sklearn.tree.tree.ExtraTreeRegressor
        state = obj.__getstate__()
        return {
            '__mlspl_type': [type(obj).__module__, type(obj).__name__],
            'state': state
        }

    @classmethod
    def decode(cls,obj):
        from sklearn.tree.tree import ExtraTreeRegressor
        state = obj['state']
        t = ExtraTreeRegressor.__new__(ExtraTreeRegressor)
        t.__setstate__(state)
        return t