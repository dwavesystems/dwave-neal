# dwave_SAGE

[![Build Status](https://travis-ci.org/dwavesystems/dwave_SAGE.svg?branch=master)](https://travis-ci.org/dwavesystems/dwave_SAGE)
[![Build status](https://ci.appveyor.com/api/projects/status/ip1j34wt9s9xsvnm?svg=true)](https://ci.appveyor.com/project/wbernoudy/dwave-sage)
[![Documentation Status](https://readthedocs.org/projects/dwave-sage/badge/?version=latest)](http://dwave-sage.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/dwavesystems/dwave_SAGE/badge.svg?branch=master)](https://coveralls.io/github/dwavesystems/dwave_SAGE?branch=master)

An implementation of a simulated annealing sampler for general Ising model graphs in C++ with a [dimod][1] Python wrapper.

For documentation, see <http://dwave-sage.readthedocs.io/en/latest/>.

## Installation

To install from PyPI:

```bash
$ pip install dwave_sage
```

To install from this repo, buliding the C++ from source:

```bash
$ git clone https://github.com/dwavesystems/dwave_SAGE.git
$ cd dwave_SAGE
$ pip install -r requirements.txt .
```

## Usage
```python
>>> from dwave_sage import DWaveSAGeSampler
>>> sampler = DWaveSAGeSampler()
```

`sampler` shares the sampler API from [dimod][1] and can be used accordingly.

```python
>>> h = {0: -1, 1: -1}
>>> J = {(0, 1): -1}
>>> response = sampler.sample_ising(h, J, samples=1)
>>> list(response.samples())
[{0: 1, 1: 1}]
```

[1]: https://dwavesystems.github.io/dimod/index.html
