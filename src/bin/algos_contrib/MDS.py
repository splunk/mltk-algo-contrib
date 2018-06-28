#!/usr/bin/env python

from sklearn.manifold import MDS as _MDS

from base import BaseAlgo, TransformerMixin
from codec import codecs_manager
from util.param_util import convert_params

from util import df_util

class MDS(TransformerMixin, BaseAlgo):

    def __init__(self, options):
        self.handle_options(options)
        out_params = convert_params(
            options.get('params', {}),
            ints=['k', 'max_iter', 'n_init', 'n_jobs'],
            floats=['eps'],
            bools=['metric'],
            aliases={'k': 'n_components'}
        )

        if 'max_iter' not in out_params:
            out_params.setdefault('max_iter', 300)

        if 'n_init' not in out_params:
            out_params.setdefault('n_init', 4)

        if 'n_jobs' not in out_params:
            out_params.setdefault('n_jobs', 1)

        if 'eps' not in out_params:
            out_params.setdefault('eps', 0.001)

        if 'metric' not in out_params:
            out_params.setdefault('metric', True)

        self.estimator = _MDS(**out_params)

    def rename_output(self, default_names, new_names):
        if new_names is None:
            new_names = 'MDS'
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
        codecs_manager.add_codec('algos_contrib.MDS', 'MDS', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.manifold.MDS', 'MDS', SimpleObjectCodec)
