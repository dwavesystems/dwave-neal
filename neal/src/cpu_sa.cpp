// Copyright 2018 D-Wave Systems Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// ===========================================================================

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctime>
#include <vector>
#include <iostream>
#include <stdexcept>
#include "cpu_sa.h"

// xorshift128+ as defined https://en.wikipedia.org/wiki/Xorshift#xorshift.2B
#define FASTRAND(rand) do {                       \
    uint64_t x = rng_state[0];                    \
    uint64_t const y = rng_state[1];              \
    rng_state[0] = y;                             \
    x ^= x << 23;                                 \
    rng_state[1] = x ^ y ^ (x >> 17) ^ (y >> 26); \
    rand = rng_state[1] + y;                      \
} while (0)

#define RANDMAX ((uint64_t)-1L)

using namespace std;

uint64_t rng_state[2]; // this holds the state of the rng

// Returns the energy delta from flipping variable at index `var`
// @param var the index of the variable to flip
// @param state the current state of all variables
// @param h vector of h or field value on each variable
// @param degrees the degree of each variable
// @param neighbors lists of the neighbors of each variable, such that 
//     neighbors[i][j] is the jth neighbor of variable i.
// @param neighbour_couplings same as neighbors, but instead has the J value.
//     neighbour_couplings[i][j] is the J value or weight on the coupling
//     between variables i and neighbors[i][j]. 
// @return delta energy
double get_flip_energy(int var, char *state, vector<double> & h, 
                      vector<int> & degrees, 
                      vector<vector<int>> & neighbors, 
                      vector<vector<double>> & neighbour_couplings) {
    double energy = h[var];
    // iterate over the neighbors of variable `var`
    for (int n_i = 0; n_i < degrees[var]; n_i++) {
        // increase `energy` by the state of the neighbor variable * the
        // corresponding coupler weight
        energy += state[neighbors[var][n_i]] * neighbour_couplings[var][n_i];
    }
    // we then multiply the entire energy by -2 * the state of `var` because
    // if s is the state of `var`, s_j is the state of the jth neighbor, and
    // w_j is the coupler weight between `var` and the jth neighbor, then
    // delta energy = sum_{j in neighbors} -2*s*w_j*s_j = 
    // -2*s*sum_{j in neighbors} w_j*s_j
    return -2 * state[var] * energy;
}

// Performs a single run of simulated annealing with the given inputs.
// @param state a signed char array where each char holds the state of a
//        variable. Note that this can be unitialized as it will be randomly
//        set at the beginning of this function.
// @param h vector of h or field value on each variable
// @param degrees the degree of each variable
// @param neighbors lists of the neighbors of each variable, such that 
//        neighbors[i][j] is the jth neighbor of variable i. Note
// @param neighbour_couplings same as neighbors, but instead has the J value.
//        neighbour_couplings[i][j] is the J value or weight on the coupling
//        between variables i and neighbors[i][j]. 
// @param sweeps_per_beta The number of sweeps to perform at each beta value.
//        Total number of sweeps is `sweeps_per_beta` * length of
//        `beta_schedule`.
// @param beta_schedule A list of the beta values to run `sweeps_per_beta`
//        sweeps at.
// @return Nothing, but `state` now contains the result of the run.
void simulated_annealing_run(char* state, vector<double>& h, 
                               vector<int>& degrees, 
                               vector<vector<int>>& neighbors, 
                               vector<vector<double>>& neighbour_couplings,
                               int sweeps_per_beta,
                               vector<double> beta_schedule) {
    const int num_vars = h.size();

    // this double array will hold the delta energy for every variable
    // delta_energy[v] is the delta energy for variable `v`
    double *delta_energy = (double*)malloc(num_vars * sizeof(double));

    // start with a random state
    uint64_t rand; // this will hold the value of the rng
    for (int var = 0; var < num_vars; var++) {
        int spin_mod = var % 64;
        if (spin_mod == 0) {
            // get a new 64 bit number
            FASTRAND(rand);
        }
        // rand >> spin_mod) & 1 gets a bit from the random number
        // 2*bit - 1 turns the random bit into +-1
        state[var] = 2*((rand >> spin_mod) & 1) - 1;
    }

    // build the delta_energy array by getting the delta energy for each
    // variable
    for (int var = 0; var < num_vars; var++) {
        delta_energy[var] = get_flip_energy(var, state, h, degrees, 
                                             neighbors, neighbour_couplings);
    }

    bool flip_spin;
    // perform the sweeps
    for (int beta_idx = 0; beta_idx < (int)beta_schedule.size(); beta_idx++) {
        // get the beta value for this sweep
        const double beta = beta_schedule[beta_idx];
        for (int sweep = 0; sweep < sweeps_per_beta; sweep++) {

            // this threshold will allow us to skip the metropolis update for
            // variables that have an extremely low chance of getting flipped
            const double threshold = 23 / beta;

            for (int var = 0; var < num_vars; var++) {
                // if the delta_energy for the variable is greater than
                // `threshold`, then we know exp(-delta energy*beta) < 1.1e-10,
                // (because exp(`threshold` * beta) = exp(-23) = 1.026e-10)
                // meaning there is less than 1 in 1e9 chance that the spin
                // will be accepted. in other words, we can safely ignore it.
                if (delta_energy[var] >= threshold) continue;

                flip_spin = false;

                if (delta_energy[var] <= 0.0) {
                    // automatically accept any flip that results in a lower 
                    // energy
                    flip_spin = true;
                }
                else {
                    // get a random number, storing it in rand
                    FASTRAND(rand); 
                    // accept the flip if exp(delta_energy*beta) > random(0, 1)
                    if (exp(-delta_energy[var]*beta) * RANDMAX > rand) {
                        flip_spin = true;
                    }
                }

                if (flip_spin) {
                    // since we have accepted the spin flip of variable `var`, 
                    // we need to adjust the delta energies of all the 
                    // neighboring variables
                    const char multiplier = 4 * state[var];
                    // iterate over the neighbors of `var`
                    for (int n_i = 0; n_i < degrees[var]; n_i++) {
                        int neighbor = neighbors[var][n_i];
                        // adjust the delta energy by 
                        // 4 * `var` state * coupler weight * neighbor state
                        // the 4 is because the original contribution from 
                        // `var` to the neighbor's delta energy was
                        // 2 * `var` state * coupler weight * neighbor state,
                        // so since we are flipping `var`'s state, we need to 
                        // multiply it again by 2 to get the full offset.
                        delta_energy[neighbor] += multiplier * 
                            neighbour_couplings[var][n_i] * state[neighbor];
                    }

                    // now we just need to flip its state and negate its delta 
                    // energy
                    state[var] *= -1;
                    delta_energy[var] *= -1;
                }
            }
        }
    }

    free(delta_energy);
}

// Returns the energy of a given state and problem
// @param state a char array containing the spin state to compute the energy of
// @param h vector of h or field value on each variable
// @param coupler_starts an int vector containing the variables of one side of
//        each coupler in the problem
// @param coupler_ends an int vector containing the variables of the other side 
//        of each coupler in the problem
// @param coupler_weights a double vector containing the weights of the 
//        couplers in the same order as coupler_starts and coupler_ends
// @return A double corresponding to the energy for `state` on the problem
//        defined by h and the couplers passed in
double get_state_energy(char* state, vector<double> h, 
                        vector<int> coupler_starts, vector<int> coupler_ends, 
                        vector<double> coupler_weights) {
    double energy = 0.0;
    // sum the energy due to local fields on variables
    for (unsigned int var = 0; var < h.size(); var++) {
        energy += state[var] * h[var];
    }
    // sum the energy due to coupling weights
    for (unsigned int c = 0; c < coupler_starts.size(); c++) {
        energy += state[coupler_starts[c]] * coupler_weights[c] * 
                                                        state[coupler_ends[c]];
    }
    return energy;
}

// Perform simulated annealing on a general problem
// @param states a char array of size num_samples * number of variables in the
//        problem. Will be overwritten by this function as samples are filled
//        in.
// @param num_samples the number of samples to get.
// @param h vector of h or field value on each variable
// @param coupler_starts an int vector containing the variables of one side of
//        each coupler in the problem
// @param coupler_ends an int vector containing the variables of the other side 
//        of each coupler in the problem
// @param coupler_weights a double vector containing the weights of the couplers
//        in the same order as coupler_starts and coupler_ends
// @param sweeps_per_beta The number of sweeps to perform at each beta value.
//        Total number of sweeps is `sweeps_per_beta` * length of
//        `beta_schedule`.
// @param beta_schedule A list of the beta values to run `sweeps_per_beta`
//        sweeps at.
// @return A double vector containing the energies of all the states that were
//         written to `states`.
vector<double> general_simulated_annealing(char* states, const int num_samples,
                                           vector<double> h, 
                                           vector<int> coupler_starts, 
                                           vector<int> coupler_ends, 
                                           vector<double> coupler_weights,
                                           int sweeps_per_beta,
                                           vector<double> beta_schedule,
                                           uint64_t seed) {

    // TODO 
    // assert len(states) == num_samples*num_vars*sizeof(char)
    // assert len(coupler_starts) == len(coupler_ends) == len(coupler_weights)
    // assert max(coupler_starts + coupler_ends) < num_vars
    
    // the number of variables in the problem
    const int num_vars = h.size();
    if (!((coupler_starts.size() == coupler_ends.size()) &&
                (coupler_starts.size() == coupler_weights.size()))) {
        throw runtime_error("coupler vectors have mismatched lengths");
    }
    
    // set the seed of the RNG
    // note that xorshift+ requires a non-zero seed
    if (seed == 0) seed = RANDMAX;
    rng_state[0] = seed;
    rng_state[1] = 0;

    // degrees will be a vector of the degrees of each variable
    vector<int> degrees(num_vars, 0);
    // neighbors is a vector of vectors, such that neighbors[i][j] is the jth
    // neighbor of variable i
    vector<vector<int>> neighbors(num_vars);
    // neighbour_couplings is another vector of vectors with the same structure
    // except neighbour_couplings[i][j] is the weight on the coupling between i
    // and its jth neighbor
    vector<vector<double>> neighbour_couplings(num_vars);

    // this will store the resulting energy after getting each sample
    vector<double> energies(num_samples);

    // build the degrees, neighbors, and neighbour_couplings vectors by
    // iterating over the inputted coupler vectors
    for (unsigned int cplr = 0; cplr < coupler_starts.size(); cplr++) {
        int u = coupler_starts[cplr];
        int v = coupler_ends[cplr];

        if ((u < 0) || (v < 0) || (u >= num_vars) || (v >= num_vars)) {
            throw runtime_error("coupler indexes contain an invalid variable");
        }

        // add v to u's neighbors list and vice versa
        neighbors[u].push_back(v);
        neighbors[v].push_back(u);
        // add the weights
        neighbour_couplings[u].push_back(coupler_weights[cplr]);
        neighbour_couplings[v].push_back(coupler_weights[cplr]);

        // increase the degrees of both variables
        degrees[u]++;
        degrees[v]++;
    }

    // get the simulated annealing samples
    for (int sample = 0; sample < num_samples; sample++) {
        // states is a giant spin array that will hold the resulting states for
        // all the samples, so we need to get the location inside that vector
        // where we will store the sample for this sample
        char *state = states + sample*num_vars;
        // then do the actual sample. this function will modify state, storing
        // the sample there
        simulated_annealing_run(state, h, degrees, 
                                neighbors, neighbour_couplings, 
                                sweeps_per_beta, beta_schedule);
        // compute the energy of the sample and store it in `energies`
        energies[sample] = get_state_energy(state, h, coupler_starts, 
                                         coupler_ends, coupler_weights);
    }

    // finally, return the energies vector (note that `states` should contain
    // all the computed samples)
    return energies;
}
