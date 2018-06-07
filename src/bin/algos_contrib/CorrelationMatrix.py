from base import BaseAlgo


class CorrelationMatrix(BaseAlgo):
    """Compute and return a correlation matrix."""

    def __init__(self, options):
        """Check for valid correlation type, and save it to an attribute on self."""

        feature_variables = options.get('feature_variables', {})
        target_variable = options.get('target_variable', {})

        if len(feature_variables) == 0:
            raise RuntimeError('You must supply one or more fields')

        if len(target_variable) > 0:
            raise RuntimeError('CorrelationMatrix does not support the from clause')

        valid_methods = ['spearman', 'kendall', 'pearson']

        # Check to see if parameters exist
        params = options.get('params', {})

        # Check if method is in parameters in search
        if 'method' in params:
            if params['method'] not in valid_methods:
                error_msg = 'Invalid value for method: must be one of {}'.format(
                    ', '.join(valid_methods))
                raise RuntimeError(error_msg)

            # Assign method to self for later usage
            self.method = params['method']

        # Assign default method and ensure no other parameters are present
        else:
            # Default method for correlation
            self.method = 'pearson'

            # Check for bad parameters
            if len(params) > 0:
                raise RuntimeError('The only valid parameter is method.')

    def fit(self, df, options):
        """Compute the correlations and return a DataFrame."""

        # df contains all the search results, including hidden fields
        # but the requested requested are saved as self.feature_variables
        requested_columns = df[self.feature_variables]

        # Get correlations
        correlations = requested_columns.corr(method=self.method)

        # Reset index so that all the data are in columns
        # (this is necessary for the corr method)
        output_df = correlations.reset_index()

        return output_df
