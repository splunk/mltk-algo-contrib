import numpy as np
from scipy.signal import savgol_filter

from base import BaseAlgo
from util.param_util import convert_params
from util import df_util


class SavgolFilter(BaseAlgo):

    def __init__(self, options):
        # set parameters
        params = options.get('params', {})
        out_params = convert_params(
            params,
            ints=['window_length', 'polyorder', 'deriv']
        )

        # set defaults for parameters
        if 'window_length' in out_params:
            self.window_length = out_params['window_length']
        else:
            self.window_length = 5

        if 'polyorder' in out_params:
            self.polyorder = out_params['polyorder']
        else:
            self.polyorder = 2

        if 'deriv' in out_params:
            self.deriv = out_params['deriv']
        else:
            self.deriv = 0

    def fit(self, df, options):
        X = df.copy()
        X, nans, columns = df_util.prepare_features(X, self.feature_variables)

        def f(x):
            return savgol_filter(x, self.window_length, self.polyorder, self.deriv)

        y_hat = np.apply_along_axis(f, 0, X)

        names = ['SG_%s' % col for col in columns]
        output_df = df_util.create_output_dataframe(y_hat, nans, names)
        df = df_util.merge_predictions(df, output_df)

        return df