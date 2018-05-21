""" Utility methods for use in testing."""
from inspect import getargspec
import ConfigParser

from base import BaseAlgo


def assert_signatures(algo):
    """Asserts that the signature of algorithm's methods adhere to the API.

    Args:
        algo (class): a custom algorithm class to check.

    Raises:
        AssertionError
    """
    assert issubclass(algo, BaseAlgo), 'Algorithms must inherit from BaseAlgo.'
    assert_method_signature(algo, '__init__', ['self', 'options'])
    assert_method_signature(algo, 'fit', ['self', 'df', 'options'])
    assert_method_signature(algo, 'partial_fit', ['self', 'df' 'options'])
    assert_method_signature(algo, 'apply', ['self', 'df', 'options'])
    assert_method_signature(algo, 'summary', ['self', 'options'])
    assert_method_signature(algo, 'reigster_codecs', [])


def assert_method_signature(algo, method, args):
    method = algo.__dict__.get(method)
    if method and callable(method):
        found_args = getargspec(method).args
        msg = 'Method {} has signature: {} - but should have {}'.format(method, args, found_args)
        assert found_args == args, msg


def assert_registered(algo):
    config = ConfigParser.RawConfigParser()
    # File path relative to the directory that tox.ini is in.
    config.read('src/default/algos.conf')
    assert config.has_section(algo.__name__)
