import pandas as pd
from sklearn.linear_model import OrthogonalMatchingPursuit as _OrthogonalMatchingPursuit
from base import RegressorMixin, BaseAlgo
from util.param_util import convert_params
from util import df_util


class OrthogonalMatchingPursuit(RegressorMixin, BaseAlgo):
    def __init__(self, options):
        self.handle_options(options)

        params = options.get('params', {})
        out_params = convert_params(
            params,
            floats=['tol'],
            strs=['kernel'],
            ints=['n_nonzero_coefs'],
            bools=['fit_intercept', 'normalize'],
        )

        self.estimator = _OrthogonalMatchingPursuit(**out_params)

    def summary(self, options):
        if len(options) != 2:  # only model name and mlspl_limits
            raise RuntimeError('"%s" models do not take options for summarization' % self.__class__.__name__)
        df = pd.DataFrame({'feature': self.columns,
                           'coefficient': self.estimator.coef_.ravel()})
        idf = pd.DataFrame({'feature': ['_intercept'],
                            'coefficient': [self.estimator.intercept_]})
        return pd.concat([df, idf])

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec
        from codec import codecs_manager
        codecs_manager.add_codec('algos_contrib.OrthogonalMatchingPursuit', 'OrthogonalMatchingPursuit', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.linear_model.omp', 'OrthogonalMatchingPursuit', SimpleObjectCodec)

