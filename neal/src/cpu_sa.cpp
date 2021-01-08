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

#include <math.h>
#include <vector>
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

// this holds the state of our thread-safe/local RNG
thread_local uint64_t rng_state[2];

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
double get_flip_energy(
    int var,
    char *state,
    const vector<double>& h,
    const vector<int>& degrees,
    const vector<vector<int>>& neighbors,
    const vector<vector<double>>& neighbour_couplings
) {
    double energy = h[var];
    // iterate over the neighbors of variable `var`
    for (int n_i = 0; n_i < degrees[var]; n_i++) {
        // increase `energy` by the state of the neighbor variable * the
        // corresponding coupler weight
        energy += state[neighbors[var][n_i]] * neighbour_couplings[var][n_i];
    }
    // the value of the variable `energy` is now equal to the sum of the
    // coefficients of `var`.  we then multiply this by -2 * the state of `var`
    // because the energy delta is given by: (x_i_new - x_i_old) * sum(coefs),
    // and (x_i_new - x_i_old) = -2 * x_i_old
    return -2 * state[var] * energy;
}

// Performs a single run of simulated annealing with the given inputs.
// @param state a signed char array where each char holds the state of a
//        variable. Note that this will be used as the initial state of the
//        run.
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
void simulated_annealing_run(
    char* state,
    const vector<double>& h,
    const vector<int>& degrees,
    const vector<vector<int>>& neighbors,
    const vector<vector<double>>& neighbour_couplings,
    const int sweeps_per_beta,
    const vector<double>& beta_schedule
) {
    const int num_vars = h.size();

    // this double array will hold the delta energy for every variable
    // delta_energy[v] is the delta energy for variable `v`
    double *delta_energy = (double*)malloc(num_vars * sizeof(double));

    uint64_t rand; // this will hold the value of the rng

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
            // variables that have zero chance of getting flipped.
            // our RNG generates 64 bit integers, so we have a resolution of
            // 1 / 2^64. since log(1 / 2^64) = -44.361, if the delta energy is
            // greater than 44.361 / beta, then we can safely skip computing
            // the probability.
            const double threshold = 44.36142 / beta;

            for (int var = 0; var < num_vars; var++) {
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
                    // accept the flip if exp(-delta_energy*beta) > random(0, 1)
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
double get_state_energy(
    char* state,
    const vector<double>& h,
    const vector<int>& coupler_starts,
    const vector<int>& coupler_ends,
    const vector<double>& coupler_weights
) {
    double energy = 0.0;
    // sum the energy due to local fields on variables
    for (unsigned int var = 0; var < h.size(); var++) {
        energy += state[var] * h[var];
    }
    // sum the energy due to coupling weights
    for (unsigned int c = 0; c < coupler_starts.size(); c++) {
        energy += state[coupler_starts[c]] * coupler_weights[c] * state[coupler_ends[c]];
    }
    return energy;
}

// Perform simulated annealing on a general problem
// @param states a char array of size num_samples * number of variables in the
//        problem. Will be overwritten by this function as samples are filled
//        in. The initial state of the samples are used to seed the simulated
//        annealing runs.
// @param energies a double array of size num_samples. Will be overwritten by
//        this function as energies are filled in.
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
// @param interrupt_callback A function that is invoked between each run of simulated annealing
//        if the function returns True then it will stop running.
// @param interrupt_function A pointer to contents that are passed to interrupt_callback.
// @return the number of samples taken. If no interrupt occured, will equal num_samples.
int general_simulated_annealing(
    char* states,
    double* energies,
    const int num_samples,
    const vector<double> h,
    const vector<int> coupler_starts,
    const vector<int> coupler_ends,
    const vector<double> coupler_weights,
    const int sweeps_per_beta,
    const vector<double> beta_schedule,
    const uint64_t seed,
    callback interrupt_callback,
    void * const interrupt_function
) {
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
    rng_state[0] = seed ? seed : RANDMAX;
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
    int sample = 0;
    while (sample < num_samples) {
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

        sample++;

        // if interrupt_function returns true, stop sampling
        if (interrupt_function && interrupt_callback(interrupt_function)) break;
    }

    // return the number of samples we actually took
    return sample;
}
