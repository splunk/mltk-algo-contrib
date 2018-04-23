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

Thirdly, create a virtualenv, and install the requirements.txt:

```bash
virtualenv virtenv
source virtenv/bin/activate
```

- Add your new algorithm(s) to `src/bin/algos`.
- Add a new stanza to `src/default/algos.conf`

## Running Tests

Inside a python interpreter, import & call:

```python
from test import run_tests
run_tests()
```

Additionally, you can run individual ad-hoc tests:

```python
# Add the MLTK to our sys.path
from link_mltk import add_mltk
add_mltk()

import pandas as pd

# Import our algorithm class
from algos.ExampleAlgo import ExampleAlgo

# Use utilties to catch common mistakes
from test.util import check_signatures
check_signatures(ExampleAlgo)

# Try loading a DataFrame and calling fit
df = pd.read_csv('test/data/iris.csv')
options = {'feature_variables': ['petal_length'], 'target_variable': 'species'}
algo_instance = ExampleAlgo(options)
algo_instance.fit(df, options)
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
