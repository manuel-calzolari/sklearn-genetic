.. -*- mode: rst -*-

|PyPi|_ |Conda|_ |ReadTheDocs|_

.. |PyPi| image:: https://img.shields.io/pypi/v/sklearn-genetic?style=flat-square
.. _PyPi: https://pypi.org/project/sklearn-genetic

.. |Conda| image:: https://img.shields.io/conda/v/conda-forge/sklearn-genetic?style=flat-square
.. _Conda: https://anaconda.org/conda-forge/sklearn-genetic

.. |ReadTheDocs| image:: https://readthedocs.org/projects/sklearn-genetic/badge/?version=latest&style=flat-square
.. _ReadTheDocs: https://sklearn-genetic.readthedocs.io/en/latest/?badge=latest

***************
sklearn-genetic
***************

**sklearn-genetic** is a genetic feature selection module for scikit-learn.

Genetic algorithms mimic the process of natural selection to search for optimal values of a function.

Installation
============

Dependencies
------------

sklearn-genetic requires:

- Python (>= 3.7)
- scikit-learn (>= 1.0)
- deap (>= 1.0.2)
- numpy
- multiprocess

User installation
-----------------

The easiest way to install sklearn-genetic is using :code:`pip`

.. code:: bash

    pip install sklearn-genetic

or :code:`conda`

.. code:: bash

    conda install -c conda-forge sklearn-genetic

Documentation
=============

Installation documentation, API reference and examples can be found on the `documentation <https://sklearn-genetic.readthedocs.io>`_.

See also
========

- `shapicant <https://github.com/manuel-calzolari/shapicant>`_, a feature selection package based on SHAP and target permutation, for pandas and Spark
