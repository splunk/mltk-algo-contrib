#!/usr/bin/env python

from sklearn.cluster import Birch as _Birch

from base import BaseAlgo, ClustererMixin
from codec import codecs_manager
from codec.codecs import BaseCodec
from codec.flatten import flatten, expand
from util import df_util
from util.param_util import convert_params


class BirchCodec(BaseCodec):
    @classmethod
    def encode(cls, obj):
        """Birch has circular references and must be flattened."""
        flat_obj, refs = flatten(obj)

        return {
            '__mlspl_type': [type(obj).__module__, type(obj).__name__],
            'dict': flat_obj.__dict__,
            'refs': refs,
        }

    @classmethod
    def decode(cls, obj):
        import sklearn.cluster

        m = sklearn.cluster._birch.Birch.__new__(sklearn.cluster._birch.Birch)
        m.__dict__ = obj['dict']

        return expand(m, obj['refs'])


class Birch(ClustererMixin, BaseAlgo):
    def __init__(self, options):
        self.handle_options(options)

        out_params = convert_params(
            options.get('params', {}), ints=['k'], aliases={'k': 'n_clusters'}
        )

        self.estimator = _Birch(**out_params)

    def apply(self, df, options):
        """Apply is overriden to make prediction on chunks of 10000 rows."""
        func = super(self.__class__, self).apply
        return df_util.apply_in_chunks(df, func, 10000, options)

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec

        codecs_manager.add_codec('sklearn.cluster._birch', 'Birch', BirchCodec)
        codecs_manager.add_codec('codec.flatten', 'Ref', SimpleObjectCodec)
        codecs_manager.add_codec('algos.Birch', 'Birch', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.cluster._birch', '_CFNode', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.cluster._birch', '_CFSubcluster', SimpleObjectCodec)
