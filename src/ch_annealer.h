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

#ifndef __DWAVE__chimera_annealer_h
#define __DWAVE__chimera_annealer_h

#include <stdint.h>
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
#include <map>

#include "common.h"
#include "annealer.h"

#define EXP_PRECISION           8.0
#define EXP_1_PRECISION         (1 / EXP_PRECISION)
#define EXP_TABSIZE             24


class chimera_annealer : public annealer {

public:
    
    /**
     * chimera_annealer class constructor
     * receives problem parameters (fields and couplers) and random seed to use
     * calls annaeler constructor for initialization
     * 
     * @param field_vec         list of field values
     * @param coupler_starts    list of start vertices of couplers
     * @param coupler_ends      list of end vertices of couplers
     * @param coupler_values    list of values of couplers
     * @param seed              random seed values (use 0 for system time)
     */
    chimera_annealer(std::vector<double> field_vec, std::vector<int> coupler_starts, std::vector<int> coupler_ends, std::vector<double> coupler_values, unsigned int seed = 0);
    
    ~chimera_annealer();

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
    void anneal(int num_samples, int num_sweeps, int sweeps_per_beta, double beta_start, double beta_end, bool use_geometric_schedule);

protected:
    
    /**
     * performs a sweep by calling sweep_taylor(sample, beta)
     * 
     * @input sample    the sample index to perform the sweep
     * @input beta      temperature inverse to use in the sweep
     */
    inline void sweep(int sample, float beta);
    
    /**
     * performs a sweep using expf for computing e^x
     * 
     * @input sample    the sample index to perform the sweep
     * @input beta      temperature inverse to use in the sweep
     */
    void sweep_expf(int sample, float beta);
    
    /**
     * performs a sweep using a fast approximation of expf
     * based on a lookup table and taylor series
     * 
     * @input sample    the sample index to perform the sweep
     * @input beta      temperature inverse to use in the sweep
     */
    void sweep_taylor(int sample, float beta);
    
    void calculate_delta_energies(int sample);
    void update_spins(int sample, int start_spin, float beta);

    void initialize(int num_samples, int num_sweeps, int sweeps_per_beta, double beta_start, double beta_end, bool use_geometric_schedule);

    void calculate_energies();

};

#endif  // __DWAVE__chimera_annealer_h
