'''
Once newer version of sklearn is used will need to change k alias from n_topics to n_components
https://stackoverflow.com/a/48121678
'''

from sklearn.decomposition import LatentDirichletAllocation as _LatentDirichletAllocation
from base import BaseAlgo, TransformerMixin
from codec import codecs_manager
from util.param_util import convert_params

class LatentDirichletAllocation(TransformerMixin, BaseAlgo):

    def __init__(self, options):
        self.handle_options(options)
        out_params = convert_params(
            options.get('params', {}),
            floats=['doc_topic_prior','learning_decay','learning_offset','perp_tol','mean_change_tol'],
            strs=['learning_method'],
            ints=['k','max_iter','batch_size','evaluate_every','total_samples','max_doc_update_iter','n_jobs','verbose','random_state'],
            aliases={'k': 'n_topics'}
        )

        self.estimator = _LatentDirichletAllocation(**out_params)

    def rename_output(self, default_names, new_names):
        if new_names is None:
            new_names = 'LDA'
        output_names = ['{}_{}'.format(new_names, i+1) for i in xrange(len(default_names))]
        return output_names

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec
        codecs_manager.add_codec('algos_contrib.LatentDirichletAllocation', 'LatentDirichletAllocation', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.decomposition.online_lda', 'LatentDirichletAllocation', SimpleObjectCodec)
