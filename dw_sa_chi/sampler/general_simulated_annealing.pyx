from libcpp.vector cimport vector
cimport numpy as np
import numpy as np

cdef extern from "cpu_sa.h":
    vector[double] general_simulated_annealing(
            char*, # giant array that holds all states
            const int, # number of sweeps
            vector[double] &, # h
            vector[int] &, # coupler starts
            vector[int] &, # coupler ends
            vector[double] &, # coupler weights
            vector[double] &, # beta schedule
            unsigned long long) # seed

def simulated_annealing(num_sweeps, h, coupler_starts, coupler_ends, 
                        coupler_weights, beta_schedule, seed):

    num_vars = len(h)
    cdef np.ndarray[char, ndim=1, mode="c"] states_numpy = \
            np.empty(num_sweeps*num_vars, dtype="b")
    cdef char* states = &states_numpy[0]
    
    energies = general_simulated_annealing(states, num_sweeps, h, 
                                           coupler_starts, coupler_ends, 
                                           coupler_weights, beta_schedule, 
                                           seed)

    annealed_states = states_numpy.reshape((num_sweeps, num_vars))

    return annealed_states, np.asarray(energies)
