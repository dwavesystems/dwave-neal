# DWSAGe

An implementation of a simulated annealing sampler for general Ising model graphs in C++ with a [dimod][1] Python wrapper.

## Installation

To install from PyPI:

```bash
$ pip install dwave_sage
```

To install from this repo, buliding the C++ from source:

```bash
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
