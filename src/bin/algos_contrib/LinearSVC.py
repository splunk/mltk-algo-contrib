#!/usr/bin/env python

from sklearn.svm import LinearSVC as _LinearSVC

from codec import codecs_manager
from base import BaseAlgo, ClassifierMixin
from util.param_util import convert_params


class LinearSVC(ClassifierMixin, BaseAlgo):

    def __init__(self, options):
        self.handle_options(options)

        out_params = convert_params(
            options.get('params', {}),
            floats=['gamma', 'C', 'tol', 'intercept_scaling'],
            ints=['random_state','max_iter'],
            strs=['penalty', 'loss', 'multi_class'],
            bools=['dual', 'fit_intercept'],
        )

        self.estimator = _LinearSVC(**out_params)

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec
        codecs_manager.add_codec('algos_contrib.LinearSVC', 'LinearSVC', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.svm.classes', 'LinearSVC', SimpleObjectCodec)
