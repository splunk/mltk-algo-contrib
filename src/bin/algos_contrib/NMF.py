from sklearn.decomposition import NMF as _NMF
from base import BaseAlgo, TransformerMixin
from codec import codecs_manager
from util.param_util import convert_params

class NMF(TransformerMixin, BaseAlgo):

    def __init__(self, options):
        self.handle_options(options)
        out_params = convert_params(
            options.get('params', {}),
            floats=['beta_loss','tol','alpha','l1_ratio'],
            strs=['init','solver'],
            ints=['k','max_iter','random_state'],
            bools=['versbose','shuffle'],
            aliases={'k': 'n_components'}
        )

        self.estimator = _NMF(**out_params)

    def rename_output(self, default_names, new_names):
        if new_names is None:
            new_names = 'NMF'
        output_names = ['{}_{}'.format(new_names, i+1) for i in xrange(len(default_names))]
        return output_names

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec
        codecs_manager.add_codec('algos_contrib.NMF', 'NMF', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.decomposition.nmf', 'NMF', SimpleObjectCodec)
