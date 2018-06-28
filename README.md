# mltk-algo-contrib

This repo contains custom algorithms for use with the [Splunk Machine Learning Toolkit](https://splunkbase.splunk.com/app/2890/). The repo itself is also a Splunk app.
Custom algorithms can be added to the Splunk Machine Learning toolkit by adhering to the [ML-SPL API](http://docs.splunk.com/Documentation/MLApp/latest/API/Introduction).
The API is a thin wrapper around machine learning estimators provided by libraries such as:
* [scikit-learn](scikit-learn.org)
* [statsmodels](http://www.statsmodels.org/).
* [scipy](https://www.scipy.org)

and custom algorithms.

Note that this repo is a collection of custom *algorithms* only, and not any libraries. Any libraries required
should only be added to live environments manually and not to this repo.

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

# Usage
This repository is contains public contributions and Splunk is not responsible for guaranteeing
the correctness or validity of the algorithms. Splunk is in no way responsible for the vetting of
the contents of contributed algorithms.

# Deploying

To use the custom algorithms in this repository, you must deploy them as a Splunk app.

There are two ways to do this.

### Manual copying

You can simple copy the following directories under src:
  * bin
  * default
  * metadata

to:
  * ${SPLUNK_HOME}/etc/apps/SA_mltk_contrib_app (you will need to create the directory first):

OR

### Build and install

#### 1. Build the app:

You will need to install tox. See [Test Prerequisites](#prereq)

```bash
tox -e package-macos        # if on Mac
tox -e package-linux        # if on Linux
```

  * The resulting gzipped tarball will be in the `target` directory (e.g. target/SA_mltk_contrib_app.tgz).
    * The location of the gzipped tarball can be overridden by `BUILD_DIR` environment variable.
  * The default app name will be `SA_mltk_contrib_app`, but this can be overridden by the `APP_NAME` environment variable.

* **NOTE**: You can run `tox -e clean` to remove the `target` directory.

#### 2. Install the tarball:

  * You can do one of the followings with the tarball from step 1:
    * Manually untar it in `${SPLUNK_HOME}/etc/apps` directory
    * Install it using the GUI:
      * https://docs.splunk.com/Documentation/AddOns/released/Overview/Singleserverinstall

# Contributing

This repository was specifically made for your contributions! See [Contributing](https://github.com/splunk/mltk-algo-contrib/blob/master/CONTRIBUTING.md) for more details.

## Developing

To start developing, you will need to have Splunk installed. If you don't, read more [here](http://docs.splunk.com/Documentation/Splunk/latest/Installation/InstallonLinux).

1. clone the repo and cd into the directory:

```bash
git clone https://github.com/splunk/mltk-algo-contrib.git
cd mltk-algo-contrib
```

2. symlink the `src` directory to the apps folder in Splunk and restart splunkd:

```bash
ln -s "$(pwd)/src" $SPLUNK_HOME/etc/apps/SA_mltk_contrib_app
$SPLUNK_HOME/bin/splunk restart
```
  * _This will eliminate the need to deploy the app to test changes._

3. Add your new algorithm(s) to `src/bin/algos_contrib`.
  (See SVR.py for an example.)
  
4. Add a new stanza to `src/default/algos.conf`

```bash
[<your_algo>]
package=algos_contrib
```

  * **NOTE**: Due to the way configuration file layering works in Splunk,
  the package name must be overridden in each section, and not
  in the _default_ section.
    
5. Add your tests to `src/bin/algos_contrib/tests/test_<your_algo>.py`
  (See test_svr.py for an example.)

## Running Tests

<a name=prereq></a>
### Prerequisites

1. Install *tox*:
   * http://tox.readthedocs.io/en/latest/install.html
   ```bash
   pip install tox
   ```

2. Install *tox-pip-extensions*:
   * https://github.com/tox-dev/tox-pip-extensions
   ```bash
   pip install tox-pip-extensions
   ```
   * **NOTE**: You only need this if you do not want to
   recreate the virtualenv(s) manually with `tox -r`
   everytime you update requirements*.txt file, but
   this is recommended for convenience.

3. You must also have the following environment variable set to your
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
tox -- -k "example"     # Run tests that has keyword 'example'
tox -- -x               # Stop after the first failure
tox -- -s               # Show stdout/stderr (i.e. disable capturing)
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
>>> from test.contrib_util import AlgoTestUtils
>>> AlgoTestUtils.assert_algo_basic(ExampleAlgo, serializable=False)
```

### Package/File Naming

Files and packages under _test_ directory should avoid having names
that conflict with files or directories directly under:
```bash
$SPLUNK_HOME/etc/apps/Splunk_ML_Toolkit/bin
```

## Pull requests

Once you've finished what you're adding, make a pull request.

## Bugs? Issues?

Please file issues with any information that might be needed to:
 - reproduce what you're experiencing
 - understand the problem fully

# License

The algorithms hosted, as well as the app itself, is licensed under the permissive Apache 2.0 license.

**Any additions to this repository must be under one of these licenses:**
 - MIT
 - BSD
 - Apache 2.0
