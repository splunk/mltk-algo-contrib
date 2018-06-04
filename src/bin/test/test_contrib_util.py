import mock
import io
import pandas as pd
import pytest
import sys

from base import BaseAlgo
from util.base_util import MLSPLNotImplementedError

from contrib_util import AlgoTestUtils


@pytest.fixture
def min_algo_cls():
    class MinimalAlgo(BaseAlgo):
        pass
    return MinimalAlgo


@pytest.fixture
def serializable_algo_cls():
    class SerializableAlgo(BaseAlgo):
        def __init__(self, options):
            pass

        def fit(self, df, options):
            pass

        def apply(self, df, options):
            return df

        @classmethod
        def register_codecs(cls):
            from codec.codecs import SimpleObjectCodec
            from codec import codecs_manager
            codecs_manager.add_codec('test.test_contrib_util', 'SerializableAlgo', SimpleObjectCodec)

    # Add the class to this module so that encoder and decoder can access it.
    # This is only necessary for a fixture function.  Normally, these classes will be defined within a module.
    setattr(sys.modules[__name__], 'SerializableAlgo', SerializableAlgo)
    return SerializableAlgo


mock_algo_conf = """
[MinimalAlgo]
"""


def test_assert_method_signature(min_algo_cls):
    AlgoTestUtils.assert_method_signature(min_algo_cls, 'fit', ['self', 'df', 'options'])


@mock.patch.object(AlgoTestUtils, 'get_algos_conf_fp', return_value=io.BytesIO(mock_algo_conf))
def test_assert_registered(mock_get_algos_conf_fp, min_algo_cls):
    AlgoTestUtils.assert_registered(min_algo_cls)


def test_assert_serializable(serializable_algo_cls):
    AlgoTestUtils.assert_serializable(serializable_algo_cls, input_df=pd.DataFrame({}), options={})


def test_assert_base_algo_method_signatures_default_methods(min_algo_cls):
    AlgoTestUtils.assert_base_algo_method_signatures(min_algo_cls)


def test_assert_base_algo_method_signatures_all_methods(min_algo_cls):
    AlgoTestUtils.assert_base_algo_method_signatures(min_algo_cls, required_methods=[
        '__init__',
        'fit',
        'partial_fit',
        'apply',
        'register_codecs',
    ])


def test_assert_base_algo_method_signatures_extra_methods(min_algo_cls):
    with pytest.raises(AssertionError) as e:
        extra_args = [
            'extra1',
            'extra2',
        ]
        AlgoTestUtils.assert_base_algo_method_signatures(min_algo_cls, required_methods=[
            '__init__',
            'fit',
            'partial_fit',
            'apply',
            'register_codecs',
        ] + extra_args)
    assert e.match('{}.*not in BaseAlgo'.format(extra_args))


@mock.patch.object(AlgoTestUtils, 'get_algos_conf_fp', return_value=io.BytesIO(mock_algo_conf))
def test_assert_algo_basic(mock_get_algos_conf_fp, min_algo_cls):
    AlgoTestUtils.assert_algo_basic(min_algo_cls, serializable=False)


def test_no_base_algo():
    class NoBaseAlgo(object):
        pass

    with pytest.raises(AssertionError) as e:
        AlgoTestUtils.assert_base_algo_method_signatures(NoBaseAlgo)
    assert e.match('must inherit from BaseAlgo')


def test_assert_method_signature_non_existent(min_algo_cls):
    bad_method = 'foot'
    with pytest.raises(AssertionError) as e:
        AlgoTestUtils.assert_method_signature(min_algo_cls, bad_method, ['self', 'df', 'options'])
    e.match("{}.*does not exist".format(bad_method))


def test_assert_method_signature_not_callable(min_algo_cls):
    bad_method = 'fit'

    # Make fit a property.
    min_algo_cls.fit = 'fit'

    with pytest.raises(AssertionError) as e:
        AlgoTestUtils.assert_method_signature(min_algo_cls, bad_method, ['self', 'df', 'options'])
    e.match("{}.*not callable".format(bad_method))


@mock.patch.object(AlgoTestUtils, 'get_algos_conf_fp', return_value=io.BytesIO(mock_algo_conf))
def test_assert_unregistered(mock_get_algos_conf_fp):
    class UnregisteredAlgo(BaseAlgo):
        pass

    with pytest.raises(AssertionError) as e:
        AlgoTestUtils.assert_registered(UnregisteredAlgo)
    assert e.match('{}.*not registered'.format(UnregisteredAlgo.__name__))


def test_assert_not_serializable(min_algo_cls):
    with pytest.raises(MLSPLNotImplementedError) as e:
        AlgoTestUtils.assert_serializable(min_algo_cls, input_df=pd.DataFrame({}), options={})
    assert e.match('does not support saving')


