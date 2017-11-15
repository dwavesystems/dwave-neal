# dwave_neal

An implementation of a simulated annealing sampler for general Ising model graphs in C++ with a [dimod][1] Python wrapper.

For documentation, see <http://dwave-neal.readthedocs.io/en/latest/>.

## Installation

To install from PyPI:

```bash
$ pip install dwave_neal
```

To install from this repo, buliding the C++ from source:

```bash
$ git clone https://github.com/dwavesystems/dwave_neal.git
$ cd dwave_neal
$ pip install -r requirements.txt .
```

## Usage
```python
>>> from dwave_neal import Neal
>>> sampler = Neal()
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
