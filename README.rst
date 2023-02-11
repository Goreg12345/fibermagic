==========
fibermagic
==========

.. image:: https://img.shields.io/travis/Goreg12345/fibermagic.svg
        :target: https://travis-ci.org/Goreg12345/fibermagic

.. image:: https://img.shields.io/pypi/v/fibermagic.svg
        :target: https://pypi.python.org/pypi/fibermagic


Python package for Fiber Photometry Data Analysis

* Free software: 3-clause BSD license
* Documentation: https://fibermagic.org

Features
--------

* demodulate: removes photobleaching and drift from the signal and removes artifacts

Installation for users
----------------------

To install fibermagic, run this command in your terminal:

.. code-block:: console

    $ pip install fibermagic

Installation for developers
---------------------------

To install fibermagic, along with the tools you need to develop and run tests, clone this repository and
run the following in your virtualenv in the base directory of the repository:

.. code-block:: console

    $ python3 -m pip install -e .
    $ python3 -m pip install --upgrade -r requirements_dev.txt

The first command installs fibermagic in the active environment and creates symlinkes to the source code.
The second command installs additional dependencies for testing and documentation.

To run the tests, run the following command:

.. code-block:: console

    $ python3 -m pytest

Install pre-commit hooks

.. code-block:: console

    $ pre-commit install