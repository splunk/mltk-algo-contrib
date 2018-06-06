from sklearn.svm import SVR as _SVR

from base import BaseAlgo, RegressorMixin
from util.param_util import convert_params


class SVR(RegressorMixin, BaseAlgo):

    def __init__(self, options):
        self.handle_options(options)

        params = options.get('params', {})
        out_params = convert_params(
            params,
            floats=['C', 'gamma'],
            strs=['kernel'],
            ints=['degree'],
        )

        self.estimator = _SVR(**out_params)

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec
        from codec import codecs_manager
        codecs_manager.add_codec('algos_contrib.SVR', 'SVR', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.svm.classes', 'SVR', SimpleObjectCodec)
