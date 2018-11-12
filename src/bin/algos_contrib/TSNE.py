#!/usr/bin/env python

from sklearn.manifold import TSNE as _TSNE

from base import BaseAlgo, TransformerMixin
from codec import codecs_manager
from util.param_util import convert_params

from util import df_util

class TSNE(TransformerMixin, BaseAlgo):

    def __init__(self, options):
        self.handle_options(options)
        out_params = convert_params(
            options.get('params', {}),
            ints=['k', 'n_iter'],
            floats=['perplexity', 'early_exaggeration', 'learning_rate'],
            aliases={'k': 'n_components'}
        )

        if out_params['n_components'] < 1:
            msg = 'Invalid value for k: k must be greater than or equal to 1, but found k="{}".'
            raise RuntimeError(msg.format(out_params['n_components']))

        if 'n_iter' not in out_params:
            out_params.setdefault('n_iter', 200)

        if 'perplexity' not in out_params:
            out_params.setdefault('perplexity', 30.0)

        if 'early_exaggeration' not in out_params:
            out_params.setdefault('early_exaggeration', 4.0)

        if 'learning_rate' not in out_params:
            out_params.setdefault('learning_rate', 100)

        self.estimator = _TSNE(**out_params)

    def rename_output(self, default_names, new_names):
        if new_names is None:
            new_names = 'TSNE'
        output_names = ['{}_{}'.format(new_names, i+1) for i in xrange(len(default_names))]
        return output_names

    def apply(self, df, options):
        # Make a copy of data, to not alter original dataframe
        X = df.copy()

        # Prepare the features
        X, nans, _ = df_util.prepare_features(
            X=X,
            variables=self.feature_variables,
            final_columns=self.columns,
        )

        # Call the transform method
        y_hat = self.estimator.fit_transform(X.values)

        # Assign output_name
        output_name = options.get('output_name', None)
        default_names = self.make_output_names(
            output_name=output_name,
            n_names=y_hat.shape[1],
        )
        output_names = self.rename_output(default_names, output_name)

        # Create output dataframe
        output = df_util.create_output_dataframe(
            y_hat=y_hat,
            nans=nans,
            output_names=output_names,
        )

        # Merge with original dataframe
        output = df_util.merge_predictions(df, output)
        return output

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec
        codecs_manager.add_codec('algos_contrib.TSNE', 'TSNE', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.manifold.t_sne', 'TSNE', SimpleObjectCodec)
