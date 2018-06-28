""" Utility methods for use in testing."""
import ConfigParser
import json
import os
from inspect import getargspec

import pandas as pd

from base import BaseAlgo
from codec import MLSPLDecoder, MLSPLEncoder


PACKAGE_NAME='algos_contrib'


class AlgoTestUtils(object):
    """
    Helper methods for testing algorithm implementations
    """
    @staticmethod
    def assert_method_signature(algo_cls, method_name, args):
        """
        Assert the signature of the specified method

        Args:
            algo_cls (class): a custom algorithm class to check
            method_name (str): the name of the method
            args (list): expected arguments to the named method

        Returns:
            (bool): True if the method is callable and has the specified arguments, False otherwise.

        Raises:
            AssertionError
        """
        method = getattr(algo_cls, method_name, None)
        assert method, "Method '{}' does not exist".format(method_name)
        assert callable(method), "Method '{}' is not callable".format(method_name)
        found_args = getargspec(method).args
        msg = 'Method {} has signature: {} - but should have {}'.format(method, args, found_args)
        assert found_args == args, msg

    @classmethod
    def assert_registered(cls, algo_cls):
        """
        Assert that the algorithm is registered in the algos.conf configuration file.

        Args:
            algo_cls (class): a custom algorithm class to check

        Returns:
            (bool): True if the method is registered in algos.conf file.

        Raises:
            AssertionError
        """
        config = ConfigParser.RawConfigParser()
        with cls.get_algos_conf_fp() as f:
            config.readfp(f)
        algo_name = algo_cls.__name__
        try:
            package_name = config.get(algo_name, 'package')
        except ConfigParser.NoSectionError:
            assert False, "'{}' not registered in algos.conf".format(algo_name)
        except ConfigParser.NoOptionError:
            assert False, "'{}' must override 'package' option in algos.conf".format(algo_name)

        assert package_name == PACKAGE_NAME, "The package name must be '{}'".format(PACKAGE_NAME)

    @staticmethod
    def assert_serializable(algo_cls, input_df, options):
        """
        Assert that the model created by the algorithm is serializable.

        Args:
            algo_cls (class): a custom algorithm class to check
            input_df (pandas Dataframe): input dataframe for the algorithm being tested
            options (dict): options for the fit() (and apply(), if applicable) methods of the algorithm

        Returns:
            (bool): True if the the model is serializable, False otherwise.

        Raises:
            AssertionError
        """
        assert hasattr(algo_cls, 'register_codecs')
        algo_cls.register_codecs()

        algo_inst = algo_cls(options)
        algo_inst.feature_variables = ['b', 'c']
        algo_inst.target_variable = 'a'
        algo_inst.fit(input_df.copy(), options)

        encoded = json.dumps(algo_inst, cls=MLSPLEncoder)
        decoded = json.loads(encoded, cls=MLSPLDecoder)

        orig_y = algo_inst.apply(input_df.copy(), options)
        decoded_y = decoded.apply(input_df.copy(), options)
        pd.util.testing.assert_frame_equal(orig_y, decoded_y)

    @classmethod
    def assert_base_algo_method_signatures(cls, algo_cls, required_methods=None):
        """
        Assert that the signatures of algorithm's methods adhere to the API.

        Args:
            algo_cls (class): a custom algorithm class to check.
            required_methods (list): list of required method names.
                                     '__init__' and 'fit' are always required, so
                                     they do not need to be included.


        Returns:
            (bool): True if the methods adhere to the API, False otherwise.

        Raises:
            AssertionError
        """
        method_args_map = {
            '__init__': ['self', 'options'],
            'fit': ['self', 'df', 'options'],
            'partial_fit': ['self', 'df', 'options'],
            'apply': ['self', 'df', 'options'],
            'summary': ['self', 'options'],
            'register_codecs': [],
        }

        if required_methods is None:
            required_methods = []

        assert issubclass(algo_cls, BaseAlgo), 'Algorithms must inherit from BaseAlgo.'

        required_method_set = set(required_methods)
        extra_methods = required_method_set - method_args_map.viewkeys()
        assert extra_methods == set(), "'{}' not in BaseAlgo".format(", ".join(extra_methods))

        # __init__ and fit are always required.
        required_method_set.add('__init__')
        required_method_set.add('fit')

        for required_method in required_method_set:
            cls.assert_method_signature(algo_cls, required_method, method_args_map[required_method])

    @classmethod
    def assert_algo_basic(cls, algo_cls, required_methods=None, input_df=None, options=None, serializable=True):
        """
        Assert signatures of methods, registration, and serialization

        Args:
            algo_cls (class): a custom algorithm class to check.
            input_df (pandas Dataframe): input dataframe for the algorithm being tested
            options (dict): options for the fit() (and apply(), if applicable) methods of the algorithm
            serializable (bool): whether to check serializability or not.

        Returns:
            (bool): True if the methods adhere to the API, False otherwise.

        Raises:
            AssertionError
        """
        cls.assert_base_algo_method_signatures(algo_cls, required_methods)
        cls.assert_registered(algo_cls)
        if serializable:
            # The input and options are required for serializability test.
            assert input_df is not None
            assert options is not None
            cls.assert_serializable(algo_cls, input_df, options)

    @staticmethod
    def get_algos_conf_fp():
        """
        Get a reference (pointer) to algos.conf file open for read

        This method mainly exists to aid testing.

        Returns:
            (File): algos.conf file pointer
        """
        algos_file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'default', 'algos.conf')
        return open(algos_file_path)


