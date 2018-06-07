import numpy as np
from sklearn.cluster import AgglomerativeClustering as AgClustering
from sklearn.metrics import silhouette_samples

from base import BaseAlgo
from util.param_util import convert_params
from util import df_util


class AgglomerativeClustering(BaseAlgo):
    """Use scikit-learn's AgglomerativeClustering algorithm to cluster data."""

    def __init__(self, options):

        feature_variables = options.get('feature_variables', {})
        target_variable = options.get('target_variable', {})

        # Ensure fields are present
        if len(feature_variables) == 0:
            raise RuntimeError('You must supply one or more fields')

        # No from clause allowed
        if len(target_variable) > 0:
            raise RuntimeError('AgglomerativeClustering does not support the from clause')

        # Convert params & alias k to n_clusters
        params = options.get('params', {})
        out_params = convert_params(
            params,
            ints=['k'],
            strs=['linkage', 'affinity'],
            aliases={'k': 'n_clusters'}
        )

        # Check for valid linkage
        if 'linkage' in out_params:
            valid_linkage = ['ward', 'complete', 'average']
            if out_params['linkage'] not in valid_linkage:
                raise RuntimeError('linkage must be one of: {}'.format(', '.join(valid_linkage)))

        # Check for valid affinity
        if 'affinity' in out_params:
            valid_affinity = ['l1', 'l2', 'cosine', 'manhattan',
                              'precomputed', 'euclidean']

            if out_params['affinity'] not in valid_affinity:
                raise RuntimeError('affinity must be one of: {}'.format(', '.join(valid_affinity)))

        # Check for invalid affinity & linkage combination
        if 'linkage' in out_params and 'affinity' in out_params:
            if out_params['linkage'] == 'ward':
                if out_params['affinity'] != 'euclidean':
                    raise RuntimeError('ward linkage (default) must use euclidean affinity (default)')

        # Initialize the estimator
        self.estimator = AgClustering(**out_params)

    def fit(self, df, options):
        """Do the clustering & merge labels with original data."""
        # Make a copy of the input data
        X = df.copy()

        # Use the df_util prepare_features method to
        # - drop null columns & rows
        # - convert categorical columns into dummy indicator columns
        # X is our cleaned data, nans is a mask of the null value locations
        X, nans, columns = df_util.prepare_features(X, self.feature_variables)

        # Do the actual clustering
        y_hat = self.estimator.fit_predict(X.values)

        # attach silhouette coefficient score for each row
        silhouettes = silhouette_samples(X, y_hat)

        # Combine the two arrays, and transpose them.
        y_hat = np.vstack([y_hat, silhouettes]).T

        # Assign default output names
        default_name = 'cluster'

        # Get the value from the as-clause if present
        output_name = options.get('output_name', default_name)

        # There are two columns - one for the labels, for the silhouette scores
        output_names = [output_name, 'silhouette_score']

        # Use the predictions & nans-mask to create a new dataframe
        output_df = df_util.create_output_dataframe(y_hat, nans, output_names)

        # Merge the dataframe with the original input data
        df = df_util.merge_predictions(df, output_df)
        return df
