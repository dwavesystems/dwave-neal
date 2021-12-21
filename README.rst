.. image:: https://img.shields.io/pypi/v/dwave-neal.svg
    :target: https://pypi.org/project/dwave-neal

.. image:: https://codecov.io/gh/dwavesystems/dwave-neal/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/dwavesystems/dwave-neal

.. image:: https://readthedocs.com/projects/d-wave-systems-dwave-neal/badge/?version=latest
    :target: https://docs.ocean.dwavesys.com/projects/neal/en/latest/?badge=latest

.. image:: https://circleci.com/gh/dwavesystems/dwave-neal.svg?style=svg
    :target: https://circleci.com/gh/dwavesystems/dwave-neal

dwave-neal
==========

.. index-start-marker

An implementation of a simulated annealing sampler.

Example Usage
-------------

.. code-block:: python

    import neal

    sampler = neal.SimulatedAnnealingSampler()

    h = {0: -1, 1: -1}
    J = {(0, 1): -1}
    sampleset = sampler.sample_ising(h, J)

.. index-end-marker

Installation
------------

.. installation-start-marker

To install:

.. code-block:: bash

    pip install dwave-neal

To build from source:

.. code-block:: bash

    pip install -r requirements.txt
    python setup.py build_ext --inplace
    python setup.py install

.. installation-end-marker

License
-------

Released under the Apache License 2.0. See LICENSE file.

Contributing
============

Ocean's `contributing guide <https://docs.ocean.dwavesys.com/en/stable/contributing.html>`_
has guidelines for contributing to Ocean packages.
