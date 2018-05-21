# mltk-algo-contrib

This repo contains custom algorithms for use with the [Splunk Machine Learning Toolkit](https://splunkbase.splunk.com/app/2890/). The repo itself is also a Splunk app.
Custom algorithms can be added to the Splunk Machine Learning toolkit by adhering to the [ML-SPL API](http://docs.splunk.com/Documentation/MLApp/latest/API/Introduction).
The API is a thin wrapper around machine learning estimators provided by libraries such as [scikit-learn](scikit-learn.org) or [statsmodels](http://www.statsmodels.org/).

A comprehensive guide to using the ML-SPL API can be found [here](http://docs.splunk.com/Documentation/MLApp/latest/API/Introduction).

A very simple example:

```python
from base import BaseAlgo


class CustomAlgorithm(BaseAlgo):
    def __init__(self, options):
        # Option checking & initializations here
        pass

    def fit(self, df, options):
        # Fit an estimator to df, a pandas DataFrame of the search results
        pass

    def partial_fit(self, df, options):
        # Incrementally fit a model
        pass

    def apply(self, df, options):
        # Apply a saved model
        # Modify df, a pandas DataFrame of the search results
        return df

    @staticmethod
    def register_codecs():
        # Add codecs to the codec manager
        pass

```

# Dependencies

To use the custom algorithms contained in this app, you must also have installed:

 - [Splunk Machine Learning Toolkit](https://splunkbase.splunk.com/app/2890/) 
 - Python for Scientific Computing Add-on
    - [Linux64](https://splunkbase.splunk.com/app/2882/)
    - [Linux32](https://splunkbase.splunk.com/app/2884/)
    - [Windows64](https://splunkbase.splunk.com/app/2883/)
    - [macOS](https://splunkbase.splunk.com/app/2881/)

# Contributing

This repository was specifically made for your contributions!

## Developing

To start developing, you will need to have Splunk installed. If you don't, read more [here](http://docs.splunk.com/Documentation/Splunk/latest/Installation/InstallonLinux).

First, clone the repo:

```bash
git clone https://github.com/splunk/mltk-algo-contrib.git
```

Secondly, symlink the `src` to the apps folder in Splunk:

```bash
ln -s ./src $SPLUNK_HOME/etc/apps/SA_mltk_contrib_app
$SPLUNK_HOME/bin/splunk restart
```

Thirdly, create a virtualenv (e.g. in your home directory), and install the requirements.txt:

```bash
virtualenv $HOME/virtenv
source $HOME/virtenv/bin/activate
```

- Add your new algorithm(s) to `src/bin/algos_contrib`.
- Add a new stanza to `src/default/algos.conf`
- Add your tests to `src/bin/algos_contrib/tests/test_<your_algo>.py`
  (See test_example_algo.py for an example.)

## Running Tests

### Prerequisites

1. Install *tox*:
   * http://tox.readthedocs.io/en/latest/install.html
2. You must also have the following environment variable set to your
Splunk installation directory (e.g. /opt/splunk):
   * SPLUNK_HOME

### Using tox

To run all tests, run the following command in the root source directory:

```bash
tox
```

To run a single test, you can provide the directory or a file as a parameter:

```bash
tox src/bin/algos_contrib/tests/
tox src/bin/algos_contrib/tests/test_example_algo.py
...
```

Basically, any arguments passed to *tox* will be passed as an argument to the *pytest* command.
To pass in options, use double dashes (--):

```bash
tox -- -k "example"
tox -- -x
...
```

### Using Python REPL (Interactive Interpreter)

```python
$ python   # from src/bin directory
>>> # Add the MLTK to our sys.path
>>> from link_mltk import add_mltk
>>> add_mltk()
>>>
>>> # Import our algorithm class
>>> from algos_contrib.ExampleAlgo import ExampleAlgo
... (some warning from Splunk may show up)
>>>
>>> # Use utilities to catch common mistakes
>>> from test.util import assert_signatures
>>> assert_signatures(ExampleAlgo)
```

## Pull requests

Once you've finished what you're adding, make a pull request.

## Bugs? Issues?

Please file issues with any information that might be needed to:
 - reproduce what you're experiencing
 - understand the problem fully

# License

The algorithms hosted, as well as the app itself, is licensed under the permissive Apache 2.0 license.

**Any additions to this repository must be equally permissive in their licensing restrictions:**
 - MIT
 - BSD
 - Apache 2.0
