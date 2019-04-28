# distutils: language=c++
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
# =============================================================================

from libcpp cimport bool
from libcpp.vector cimport vector

import numpy as np
cimport numpy as np


cdef extern from "cpu_sa.h":
    ctypedef bool (*callback)(void *function)

    int general_simulated_annealing(
            char* samples,
            double* energies,
            const int num_samples,
            const vector[double] & h,
            const vector[int] & coupler_starts,
            const vector[int] & coupler_ends,
            const vector[double] & coupler_weights,
            const int sweeps_per_beta,
            const vector[double] & beta_schedule,
            const unsigned long long seed,
            callback interrupt_callback,
            void *interrupt_function) nogil


def simulated_annealing(num_samples, h, coupler_starts, coupler_ends,
                        coupler_weights, sweeps_per_beta, beta_schedule, seed,
                        np.ndarray[char, ndim=2, mode="c"] states_numpy,
                        interrupt_function=None):
    """Wraps `general_simulated_annealing` from `cpu_sa.cpp`. Accepts
    an Ising problem defined on a general graph and returns samples
    using simulated annealing.

    Parameters
    ----------
    num_samples : int
        Number of samples to get from the sampler.

    h : list(float)
        The h or field values for the problem.

    coupler_starts : list(int)
        A list of the start variable of each coupler. For a problem
        with the couplers (0, 1), (1, 2), and (3, 1), `coupler_starts`
        should be [0, 1, 3].

    coupler_ends : list(int)
        A list of the end variable of each coupler. For a problem
        with the couplers (0, 1), (1, 2), and (3, 1), `coupler_ends`
        should be [1, 2, 1].

    coupler_weights : list(float)
        A list of the J values or weight on each coupler, in the same
        order as `coupler_starts` and `coupler_ends`.

    sweeps_per_beta : int
        The number of sweeps to perform at each beta value provided in
        `beta_schedule`. The total number of sweeps per sample is
        sweeps_per_beta * len(beta_schedule).

    beta_schedule : list(float)
        A list of the different beta values to run sweeps at.

    seed : 64 bit int > 0
        The seed to use for the PRNG. Must be a positive integer
        greater than 0. If the same seed is used and the rest of the
        parameters are the same, the returned samples will be
        identical.

    states_numpy : np.ndarray[char, ndim=2, mode="c"], values in (-1, 1)
        The initial seeded states of the simulated annealing runs. Should be of
        a contiguous numpy.ndarray of shape (num_samples, num_variables).

    interrupt_function: function
        Should accept no arguments and return a bool. The function is
        called between samples and if it returns True, simulated annealing
        will return early with the samples it already has.

    Returns
    -------
    samples : numpy.ndarray
        A 2D numpy array where each row is a sample.

    energies: np.ndarray
        The energies.

    """
    num_vars = len(h)

    # in the case that we either need no samples or there are no variables,
    # we can safely return an empty array (and set energies to 0)
    if num_samples*num_vars == 0:
        annealed_states = np.empty((num_samples, num_vars), dtype=np.int8)
        return annealed_states, np.zeros(num_samples, dtype=np.double)

    # allocate ndarray for energies
    energies_numpy = np.empty(num_samples, dtype=np.float64)
    cdef double[:] energies = energies_numpy

    # explicitly convert all Python types to C while we have the GIL
    cdef char* _states = &states_numpy[0, 0]
    cdef double* _energies = &energies[0]
    cdef int _num_samples = num_samples
    cdef vector[double] _h = h
    cdef vector[int] _coupler_starts = coupler_starts
    cdef vector[int] _coupler_ends = coupler_ends
    cdef vector[double] _coupler_weights = coupler_weights
    cdef int _sweeps_per_beta = sweeps_per_beta
    cdef vector[double] _beta_schedule = beta_schedule
    cdef unsigned long long _seed = seed

    cdef void* _interrupt_function
    if interrupt_function is None:
        _interrupt_function = NULL
    else:
        _interrupt_function = <void *>interrupt_function

    with nogil:
        num = general_simulated_annealing(_states,
                                          _energies,
                                          _num_samples,
                                          _h,
                                          _coupler_starts,
                                          _coupler_ends,
                                          _coupler_weights,
                                          _sweeps_per_beta,
                                          _beta_schedule,
                                          _seed,
                                          interrupt_callback,
                                          _interrupt_function)

    # discard the noise if we were interrupted
    return states_numpy[:num], energies_numpy[:num]


cdef bool interrupt_callback(void * const interrupt_function) with gil:
    try:
        return (<object>interrupt_function)()
    except Exception:
        # if an exception occurs, treat as an interrupt
        return True
