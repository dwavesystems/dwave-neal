/**
 * Copyright 2015 D-Wave Systems Inc.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *    http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __DWAVE__annealer_h
#define __DWAVE__annealer_h

#include <map>

#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <stack>
#include <set>
#include <random>
#include <chrono>
#include <limits.h>
#include <functional>
#include <ctime>
#include <algorithm>

#include "common.h"

#define EXP_PRECISION           8.0
#define EXP_1_PRECISION         (1 / EXP_PRECISION)
#define EXP_TABSIZE             24

/**
 * Abstract Class annealer
 */
class annealer {

public:
    /**
     * annealer abstract class constructor
     * receives problem parameters (fields and couplers) and random seed to use
     * 
     * @param field_vec         list of field values
     * @param coupler_starts    list of start vertices of couplers
     * @param coupler_ends      list of end vertices of couplers
     * @param coupler_values    list of values of couplers
     * @param seed              random seed values (use 0 for system time)
     */
    annealer(std::vector<double> field_vec, std::vector<int> coupler_starts, std::vector<int> coupler_ends, std::vector<double> coupler_values, bool rearrange = false, unsigned int seed = 0);
    
    virtual ~annealer();
    
    /**
     * Perform a number of concurrent runs of simulated annealing.
     * 
     * @param num_samples               number of desired restarts (repetitions)  
     * @param num_sweeps                number of desired sweeps (per repition)
     * @param sweeps_per_beta           number of sweeps at each beta (inverse of temperature)
     * @param beta_start                initial beta
     * @param beta_end                  final beta
     * @param use_geometric_schedule    desired beta schedule type (true = geometric, false = linear)
     * 
     * @return none
     */
    virtual void anneal(int num_samples, int num_sweeps, int sweeps_per_beta, double beta_start, double beta_end, bool use_geometric_schedule) = 0;

    /**
     * returns final energies of the solutions found by SA
     * 
     * @return final energies
     */
    std::vector<double> get_energies();

    /**
     * returns final states of the solutions found by SA
     * 
     * @return final states
     */
    std::vector< std::vector< int > > get_states();

    /*
     * Print details to console.
     */
    void print_states();
    void print_energies(bool verbose=false);
    void print_energy_histogram();
    
protected:
    
    void random_state(spin_t* destination, int num);
    void calculate_delta_energies(int sample);
    void update_spins(int sample, int start_spin, float beta);

    void calculate_energies();

    /** 
     * helper function to re-arrange vertex indices based on graph bipartition
     * 
     * @param v vertex index in standard chimera representation
     * @return index of v in bipartite representation
     */
    inline int rearrange_vertex(int v);

protected:

    int num_spins;      // number of spins (vertices) in the input graph
    int num_couplers;   // number of couplers (edges) in the input graph
    float* fields;      // field values

    int* degrees;                       // to store vertex degrees
    int* neighbours = NULL;             // to store vertex neighbours
    float* neighbour_couplings = NULL;  // to store coupler values of neighbour

    float max_value;            // maximum coupler and field value in the input graph
    float max_value__1;         // 1 \over maximum coupler and field value in the input graph

    unsigned int seed;
    std::mt19937 rng;

    int num_samples = -1;               // number of repititions required
    spin_t *states = NULL;              // to store all the states 
    float *delta_energies = NULL;       // to store delta_energies for each vertex (in each sample)
    double *energies = NULL;            // to store final energy of every sample

    uint32_t *rand_seeds = NULL;

    int num_beta_steps;
    float *betas = NULL;

    float exptab[EXP_TABSIZE * (int)EXP_PRECISION];     // lookup table used for fast expf() approximation

};

#endif  // __DWAVE__annealer_h
