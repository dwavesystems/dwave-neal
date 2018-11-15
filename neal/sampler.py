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
"""
A dimod sampler_ that uses the simulated annealing algorithm.

"""
from __future__ import division

import math

from numbers import Integral
from random import randint

import dimod
import numpy as np

from six import itervalues, iteritems

import neal.simulated_annealing as sa


__all__ = ["Neal", "SimulatedAnnealingSampler"]


class SimulatedAnnealingSampler(dimod.Sampler):
    """Simulated annealing sampler.

    Also aliased as :class:`.Neal`.

    Examples:
        This example solves a simple Ising problem.

        >>> import neal
        >>> sampler = neal.SimulatedAnnealingSampler()
        >>> h = {'a': 0.0, 'b': 0.0, 'c': 0.0}
        >>> J = {('a', 'b'): 1.0, ('b', 'c'): 1.0, ('a', 'c'): 1.0}
        >>> response = sampler.sample_ising(h, J)
        >>> for sample in response:  # doctest: +SKIP
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
    """dict: A dict where keys are the keyword parameters accepted by the sampler methods
    (allowed kwargs) and values are lists of :attr:`SimulatedAnnealingSampler.properties`
    relevant to each parameter.

    See :meth:`.SimulatedAnnealingSampler.sample` for a description of the parameters.

    Examples:
        This example looks at a sampler's parameters and some of their values.

        >>> import neal
        >>> sampler = neal.SimulatedAnnealingSampler()
        >>> for kwarg in sorted(sampler.parameters):
        ...     print(kwarg)
        beta_range
        beta_schedule_type
        num_reads
        seed
        sweeps
        >>> sampler.parameters['beta_range']
        []
        >>> sampler.parameters['beta_schedule_type']
        ['beta_schedule_options']

    """

    properties = None
    """dict: A dict containing any additional information about the sampler.

    Examples:
        This example looks at the values set for a sampler property.

        >>> import neal
        >>> sampler = neal.SimulatedAnnealingSampler()
        >>> sampler.properties['beta_schedule_options']
        ('linear', 'geometric')

    """

    def __init__(self):
        # create a local copy in case folks for some reason want to modify them
        self.parameters = {'beta_range': [],
                           'num_reads': [],
                           'sweeps': [],
                           'beta_schedule_type': ['beta_schedule_options'],
                           'seed': [],
                           'interrupt_function': [],
                           'initial_states': []}
        self.properties = {'beta_schedule_options': ('linear', 'geometric')
                           }

    def sample(self, _bqm, beta_range=None, num_reads=10, sweeps=1000,
               beta_schedule_type="geometric", seed=None,
               interrupt_function=None, initial_states=None):
        """Sample from a binary quadratic model using an implemented sample method.

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                The binary quadratic model to be sampled.

            beta_range (tuple, optional):
                A 2-tuple defining the beginning and end of the beta schedule, where beta is the
                inverse temperature. The schedule is applied linearly in beta. Default range is set
                based on the total bias associated with each node.

            num_reads (int, optional, default=10):
                Each read is the result of a single run of the simulated annealing algorithm.

            sweeps (int, optional, default=1000):
                Number of sweeps or steps.

            beta_schedule_type (string, optional, default='geometric'):
                Beta schedule type, or how the beta values are interpolated between
                the given 'beta_range'. Supported values are:

                * linear
                * geometric

            seed (int, optional):
                Seed to use for the PRNG. Specifying a particular seed with a constant
                set of parameters produces identical results. If not provided, a random seed
                is chosen.

            initial_states (tuple(numpy.ndarray, dict), optional):
                A tuple where the first value is a numpy array of initial states to seed the
                simulated annealing runs, and the second is a dict defining a linear variable
                labelling.

            interrupt_function (function, optional):
                If provided, interrupt_function is called with no parameters between each sample of
                simulated annealing. If the function returns True, then simulated annealing will
                terminate and return with all of the samples and energies found so far.

        Returns:
            :obj:`dimod.Response`: A `dimod` :obj:`~dimod.Response` object.

        Examples:
            This example runs simulated annealing on a binary quadratic model with some
            different input paramters.

            >>> import dimod
            >>> import neal
            ...
            >>> sampler = neal.SimulatedAnnealingSampler()
            >>> bqm = dimod.BinaryQuadraticModel({'a': .5, 'b': -.5}, {('a', 'b'): -1}, 0.0, dimod.SPIN)
            >>> # Run with default parameters
            >>> response = sampler.sample(bqm)
            >>> # Run with specified parameters
            >>> response = sampler.sample(bqm, seed=1234, beta_range=[0.1, 4.2],
            ...                                num_reads=1, sweeps=20,
            ...                                beta_schedule_type='geometric')
            >>> # Reuse a seed
            >>> a1 = next((sampler.sample(bqm, seed=88)).samples())['a']
            >>> a2 = next((sampler.sample(bqm, seed=88)).samples())['a']
            >>> a1 == a2
            True

        """

        # if already index-labelled, just continue
        if all(v in _bqm.linear for v in range(len(_bqm))):
            bqm = _bqm
            use_label_map = False
        else:
            try:
                inverse_mapping = dict(enumerate(sorted(_bqm.linear)))
            except TypeError:
                # in python3 unlike types cannot be sorted
                inverse_mapping = dict(enumerate(_bqm.linear))
            mapping = {v: i for i, v in iteritems(inverse_mapping)}

            bqm = _bqm.relabel_variables(mapping, inplace=False)
            use_label_map = True

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

        if interrupt_function is None:
            def interrupt_function():
                return False

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
            beta_schedule = np.linspace(*beta_range, num=num_betas)
        elif beta_schedule_type == "geometric":
            # interpolate a geometric beta schedule
            beta_schedule = np.geomspace(*beta_range, num=num_betas)
        else:
            raise ValueError("Beta schedule type {} not implemented".format(beta_schedule_type))

        if seed is None:
            # pick a random seed
            seed = randint(0, (1 << 64 - 1))

        np_rand = np.random.RandomState(seed % 2**32)

        states_shape = (num_reads, num_variables)

        if initial_states is not None:
            initial_states_array, init_label_map = initial_states
            if not initial_states_array.shape == states_shape:
                raise ValueError("`initial_states` must have shape "
                                 "{}".format(states_shape))
            if init_label_map is not None:
                get_label = inverse_mapping.get if use_label_map else lambda i: i
                initial_states_array = initial_states_array[:, [init_label_map[get_label(i)] for i in range(num_variables)]]
            numpy_initial_states = np.ascontiguousarray(initial_states_array, dtype=np.int8)
        else:
            numpy_initial_states = 2*np_rand.randint(2, size=(num_reads, num_variables)).astype(np.int8) - 1

        # run the simulated annealing algorithm
        samples, energies = sa.simulated_annealing(num_reads, h,
                                                   coupler_starts, coupler_ends,
                                                   coupler_weights,
                                                   sweeps_per_beta, beta_schedule,
                                                   seed,
                                                   numpy_initial_states,
                                                   interrupt_function)
        off = bqm.spin.offset
        info = {
            "beta_range": beta_range,
            "beta_schedule_type": beta_schedule_type
        }
        response = dimod.Response.from_samples(
            samples,
            {'energy': energies+off},
            info=info,
            vartype=dimod.SPIN
        )
        
        response.change_vartype(bqm.vartype, inplace=True)
        if use_label_map:
            response.relabel_variables(inverse_mapping, inplace=True)
        
        return response


Neal = SimulatedAnnealingSampler


def _default_ising_beta_range(h, J):
    """Determine the starting and ending beta from h J

    Args:
        h (dict)

        J (dict)

    Assume each variable in J is also in h.

    We use the minimum bias to give a lower bound on the minimum energy gap, such at the
    final sweeps we are highly likely to settle into the current valley.
    """
    # Get nonzero, absolute biases
    abs_h = [abs(hh) for hh in h.values() if hh != 0]
    abs_J = [abs(jj) for jj in J.values() if jj != 0]
    abs_biases = abs_h + abs_J

    if not abs_biases:
        return [0.1, 1.0]

    # Rough approximation of min change in energy
    min_delta_energy = min(abs_biases)
   
    # Combine absolute biases by variable
    abs_bias_dict = {k: abs(v) for k, v in h.items()}
    for (k1, k2), v in J.items():
        abs_bias_dict[k1] += abs(v)
        abs_bias_dict[k2] += abs(v)

    # Find max change in energy
    max_delta_energy = sum(abs_biases) - min(abs_bias_dict.values())

    # Selecting betas based on probability of flipping a qubit
    # Hot temp: When we get max change in energy, we want at least 50% of flipping
    #   0.50 = exp(-hot_beta * max_delta_energy)
    #
    # Cold temp: Want to minimize chances of flipping. Hence, if we get the minimum change in
    #   energy, chance of flipping is set to 1%
    #   0.01 = exp(-cold_beta * min_delta_energy)
    hot_beta = np.log(2) / max_delta_energy
    cold_beta = np.log(100) / min_delta_energy

    return [hot_beta, cold_beta]
