from algos.ExampleAlgo import ExampleAlgo
from test.util import (
    assert_registered,
    assert_signatures,
)

def test_signatures():
    assert_signatures(ExampleAlgo)

def test_registration():
    assert_registered(ExampleAlgo)
