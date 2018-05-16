from libcpp.vector cimport vector

cdef extern from "cpu_sa.h":
    vector[double] general_simulated_annealing(
            char*,  # giant array that holds all states
            const int,  # number of samples
            vector[double] &,  # h
            vector[int] &,  # coupler starts
            vector[int] &,  # coupler ends
            vector[double] &,  # coupler weights
            int, # sweeps per beta
            vector[double] &,  # beta schedule
            unsigned long long)  # seed
