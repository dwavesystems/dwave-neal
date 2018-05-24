# Copyright 2018 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# ================================================================================================

from __future__ import division

import math

from numbers import Integral
from random import randint

import dimod
import numpy as np

from six import itervalues, iteritems

from neal.src import simulated_annealing


__all__ = ["Neal", "SimulatedAnnealingSampler"]


class SimulatedAnnealingSampler(dimod.Sampler):
    """Simulated annealing sampler.

    Also aliased as :class:`.Neal`.

    Examples:

        >>> import neal
        ...
        >>> sampler = neal.SimulatedAnnealingSampler()
        >>> h = {'a': 0.0, 'b': 0.0, 'c': 0.0}
        >>> J = {('a', 'b'): 1.0, ('b', 'c'): 1.0, ('a', 'c'): 1.0}
        >>> resp = sampler.sample_ising(h, J)
        >>> for sample in resp:  # doctest: +SKIP
        ...     print(sample)
        ... {'a': -1, 'b': 1, 'c': -1}
        ... {'a': -1, 'b': 1, 'c': 1}
        ... {'a': 1, 'b': 1, 'c': -1}
        ... {'a': 1, 'b': -1, 'c': -1}
        ... {'a': 1, 'b': -1, 'c': -1}
        ... {'a': 1, 'b': -1, 'c': -1}
        ... {'a': -1, 'b': 1, 'c': 1}
        ... {'a': 1, 'b': 1, 'c': -1}
        ... {'a': -1, 'b': -1, 'c': 1}
        ... {'a': -1, 'b': 1, 'c': 1}

    """

    parameters = None
    """dict: The keyword arguments accepted by SimulatedAnnealingSampler's sample methods.

    The keys are the allowed kwargs, the values are a list of the
    :attr:`SimulatedAnnealingSampler.properties` relevant to the kwarg.

    See :meth:`.SimulatedAnnealingSampler.sample` for a description of the parameters.

    Examples:

        >>> import neal
        ...
        >>> sampler = neal.SimulatedAnnealingSampler()
        >>> for kwarg in sorted(sampler.parameters):
        ...     print(kwarg)
        beta_range
        beta_schedule_type
        num_reads
        seed
        sweeps
        >>> sampler.parameters  # doctest: +SKIP
        {'beta_range': [], 'num_reads': [], 'sweeps': [], 'beta_schedule_type': ['beta_shedule_options'], 'seed': []}

    """

    properties = None
    """dict: The sampler's properties.

    Examples:
        >>> import neal
        ...
        >>> sampler = neal.SimulatedAnnealingSampler()
        >>> sampler.properties
        {'beta_shedule_options': ('linear', 'geometric')}

    """

    def __init__(self):
        # create a local copy in case folks for some reason want to modify them
        self.parameters = {'beta_range': [],
                           'num_reads': [],
                           'sweeps': [],
                           'beta_schedule_type': ['beta_shedule_options'],
                           'seed': []}
        self.properties = {'beta_shedule_options': ('linear', 'geometric')
                           }

    @dimod.decorators.bqm_index_labels
    def sample(self, bqm, beta_range=None, num_reads=10, sweeps=1000,
               beta_schedule_type="linear", seed=None):
        """Sample from low-energy states using simulated annealing.

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                The binary quadratic model to be samples.

            beta_range (tuple, optional):
                A 2-tuple defining the beginning and end of the beta schedule (beta is the
                inverse temperature). The schedule is applied linearly in beta. Default is chosen
                based on the total bias associated with each node.

            num_reads (int, optional, default=10):
                Each read is the result of a single run of the simulated annealing algorithm.

            sweeps (int, optional, default=1000):
                The number of sweeps or steps.

            beta_schedule_type (string, optional, default='lienar'):
                The beta schedule type, or how the beta values are interpolated between
                the given 'beta_range'. Default is "linear".

            seed (int, optional):
                The seed to use for the PRNG. Supplying the same seed with the rest of the same
                parameters should produce identical results. If not provided, a random seed
                is chosen.

        Returns:
            :obj:`dimod.Response`

        Examples:

            >>> import dimod
            >>> import neal
            ...
            >>> sampler = neal.SimulatedAnnealingSampler()
            >>> bqm = dimod.BinaryQuadraticModel({'a': .5, 'b': -.5}, {('a', 'b'): -1}, 0.0, dimod.SPIN)
            >>> response = sampler.sample(bqm)

        """

        # bqm is checked by decorator which also ensures that the variables are index-labelled

        # beta_range, sweeps are handled by simulated_annealing

        if not isinstance(num_reads, Integral):
            raise TypeError("'samples' should be a positive integer")
        if num_reads < 1:
            raise ValueError("'samples' should be a positive integer")

        if not (seed is None or isinstance(seed, Integral)):
            raise TypeError("'seed' should be None or a positive integer")
        if isinstance(seed, Integral) and not (0 < seed < (2**64 - 1)):
            error_msg = "'seed' should be an integer between 0 and 2^64 - 1"
            raise ValueError(error_msg)

        num_variables = len(bqm)

        # get the Ising linear biases
        linear = bqm.spin.linear
        h = [linear[v] for v in range(num_variables)]

        quadratic = bqm.spin.quadratic
        if len(quadratic) > 0:
            couplers, coupler_weights = zip(*iteritems(quadratic))
            couplers = map(lambda c: (c[0], c[1]), couplers)
            coupler_starts, coupler_ends = zip(*couplers)
        else:
            coupler_starts, coupler_ends, coupler_weights = [], [], []

        if beta_range is None:
            beta_range = _default_ising_beta_range(linear, quadratic)

        sweeps_per_beta = max(1, sweeps // 1000.0)
        num_betas = int(math.ceil(sweeps / sweeps_per_beta))
        if beta_schedule_type == "linear":
            # interpolate a linear beta schedule
            beta_step = (beta_range[1] - beta_range[0]) / float(num_betas)
            beta_schedule = [beta_range[0] + s * beta_step
                             for s in range(num_betas)]
        elif beta_schedule_type == "geometric":
            beta_step = (beta_range[1] / beta_range[0]) ** (1.0 / num_betas)
            beta_schedule = [beta_range[0] * beta_step ** i
                             for i in range(num_betas)]
        else:
            raise ValueError("Beta schedule type {} not implemented".format(beta_schedule_type))

        if seed is None:
            # pick a random seed
            seed = randint(0, (1 << 64 - 1))

        # run the simulated annealing algorithm
        samples, energies = simulated_annealing(num_reads, h,
                                                coupler_starts, coupler_ends,
                                                coupler_weights,
                                                sweeps_per_beta, beta_schedule,
                                                seed)
        off = bqm.spin.offset
        response = dimod.Response.from_matrix(samples, {'energy': [en + off for en in energies]}, vartype=dimod.SPIN)

        return response.change_vartype(bqm.vartype, inplace=True)


Neal = SimulatedAnnealingSampler


def _default_ising_beta_range(h, J):
    """Determine the starting and ending beta from h J

    Args:
        h (dict)

        J (dict)

    Assume each variable in J is also in h.

    """
    beta_range = [.1]

    sigmas = {v: abs(h[v]) for v in range(len(h))}
    for u, v in J:
        sigmas[u] += abs(J[(u, v)])
        sigmas[v] += abs(J[(u, v)])

    if len(sigmas) > 0:
        beta_range.append(2 * max(itervalues(sigmas)))
    else:
        # completely empty problem, so beta_range doesn't matter
        beta_range.append(1.0)

    return beta_range
