#!/usr/bin/env python
""" Small utility to add the MLTK bin path to the system path.
This makes it easy to import algorithms or utilities from the MLTK."""
import os
import sys


def check_splunk_home(splunk_home):
    """ Check SPLUNK_HOME and raise if not set."""
    if not splunk_home:
        raise RuntimeError('No $SPLUNK_HOME provided. Please set SPLUNK_HOME.')


def get_mltk_bin_path(splunk_home):
    """ Create the path to the MLTK bin folder."""
    check_splunk_home(splunk_home)
    mltk_path = os.path.join(splunk_home, 'etc', 'apps', 'Splunk_ML_Toolkit', 'bin')

    if not os.path.exists(mltk_path):
        raise RuntimeError('MLTK bin folder not found at {}: is MLTK installed?'.format(mltk_path))

    return mltk_path


def add_mltk():
    """ Adds MLTK bin path to sys.path """
    splunk_home = os.environ.get('SPLUNK_HOME', None)
    mltk_bin_path = get_mltk_bin_path(splunk_home)
    sys.path.insert(0, mltk_bin_path)
