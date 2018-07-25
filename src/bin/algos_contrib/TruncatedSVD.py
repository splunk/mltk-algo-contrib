from sklearn.decomposition import TruncatedSVD as _TruncatedSVD
from base import BaseAlgo, TransformerMixin
from codec import codecs_manager
from util.param_util import convert_params

class TruncatedSVD(TransformerMixin, BaseAlgo):

    def __init__(self, options):
        self.handle_options(options)
        out_params = convert_params(
            options.get('params', {}),
            floats=['tol'],
            strs=['algorithm'],
            ints=['k','n_iter','random_state'],
            aliases={'k': 'n_components'}
        )

        self.estimator = _TruncatedSVD(**out_params)

    def rename_output(self, default_names, new_names):
        if new_names is None:
            new_names = 'SVD'
        output_names = ['{}_{}'.format(new_names, i+1) for i in xrange(len(default_names))]
        return output_names

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec
        codecs_manager.add_codec('algos_contrib.TruncatedSVD', 'TruncatedSVD', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.decomposition.truncated_svd', 'TruncatedSVD', SimpleObjectCodec)
