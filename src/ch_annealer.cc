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

#include "ch_annealer.h"

using namespace std;



/**
 * chimera_annealer abstract class constructor
 * receives problem parameters (fields and couplers) and random seed to use
 * calls annaeler constructor for initialization
 * 
 * @param field_vec         list of field values
 * @param coupler_starts    list of start vertices of couplers
 * @param coupler_ends      list of end vertices of couplers
 * @param coupler_values    list of values of couplers
 * @param seed              random seed values (use 0 for system time)
 */
chimera_annealer::chimera_annealer(std::vector<double> field_vec, std::vector<int> coupler_starts, std::vector<int> coupler_ends, std::vector<double> coupler_values, unsigned int seed)
        :annealer(field_vec, coupler_starts, coupler_ends, coupler_values, seed)
{
}

chimera_annealer::~chimera_annealer()
{
}

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
void chimera_annealer::anneal(int num_samples, int num_sweeps, int sweeps_per_beta, double  beta_start, double beta_end, bool use_geometric_schedule)
{
    initialize(num_samples, num_sweeps, sweeps_per_beta, beta_start, beta_end, use_geometric_schedule);
    for (int sample = 0; sample < num_samples; sample++) { 
        calculate_delta_energies(sample);
        for(int i = 0; i < num_sweeps % sweeps_per_beta; i++) {
            sweep(sample, betas[0]);
        }

        for(int i = 1; i <= num_beta_steps; i++) {
            for (int j = 0; j < sweeps_per_beta; j++) {
                sweep(sample, betas[i]);
            }
        }
    }
    calculate_energies();
}

/**
 * performs a sweep by calling sweep_taylor(sample, beta)
 * uses normalized fields and coupler values
 * 
 * @input sample    the sample index to perform the sweep
 * @input beta      temperature inverse to use in the sweep
 */
inline void chimera_annealer::sweep(int sample, float beta)
{
    sweep_taylor(sample, beta);
}


/**
 * performs a sweep using expf for computing e^x
 * 
 * @input sample    the sample index to perform the sweep
 * @input beta      temperature inverse to use in the sweep
 */
void chimera_annealer::sweep_expf(int sample, float beta)
{

    for (int spin = 0; spin < num_spins; spin++) {
        
        bool flip;
        float x = -beta * delta_energies[spin + sample * num_spins] * max_value__1;
        float prob;

        if (x > 0.0) flip = true;
        else if (x <= -23.0) flip = false;
        else {
            prob = expf(x);
            uint32_t r;
            get_random(rand_seeds[sample], r);
            if (prob > (float)r / RAND_T_MAX) flip = true;
            else flip = false;
        }
        if (flip) {
            delta_energies[spin + sample * num_spins] = -delta_energies[spin + sample * num_spins];

            for (int i = 0; i < degrees[spin]; i++) {
                delta_energies[neighbours[i + MAX_DEGREE * (spin)] + sample * num_spins] += 
                        4 * neighbour_couplings[i + MAX_DEGREE * spin] * states[spin + sample * num_spins] * states[neighbours[i + MAX_DEGREE * (spin)] + sample * num_spins];
            }
            states[spin + sample * num_spins] *= -1;
        }
    }
}

/**
 * performs a sweep using a fast approximation of expf
 * based on a fixed lookup table and taylor series
 * uses normalized fields and coupler values
 * 
 * @input sample    the sample index to perform the sweep
 * @input beta      temperature inverse to use in the sweep
 */
void chimera_annealer::sweep_taylor(int sample, float beta)
{
    for (int spin = 0; spin < num_spins; spin++) {
        bool flip;
        float x = beta * delta_energies[spin + sample * num_spins] * max_value__1;
        float prob;

        if (x <= 0.0) flip = true;              // always flip when finding a better enegery leve
        else if (x >= 23.0) flip = false;       // never swip when delta energy is too small
        else {
            prob = 0.0;
            int n = (int)(x * EXP_PRECISION);
            x = (n * EXP_1_PRECISION) - x;
            
            prob = (FACT_1_4 + prob) * x;
            prob = (FACT_1_3 + prob) * x;
            prob = (FACT_1_2 + prob) * x;
            prob = (FACT_1_1 + prob) * x;
            prob += FACT_1_0;
            prob *= exptab[n];
            
            uint32_t r;
            get_random(rand_seeds[sample], r);
            if (prob > (float)r / RAND_T_MAX) flip = true;
            else flip = false;
        }
        if (flip) {
            delta_energies[spin + sample * num_spins] = -delta_energies[spin + sample * num_spins];

            spin_t s1 = states[spin + sample * num_spins];
            for (int i = 0; i < degrees[spin]; i++) {
                spin_t s2 = states[neighbours[i + MAX_DEGREE * (spin)] + sample * num_spins];
                delta_energies[neighbours[i + MAX_DEGREE * (spin)] + sample * num_spins] += 
                        4 * neighbour_couplings[i + MAX_DEGREE * spin] * s1 * s2;
            }
            states[spin + sample * num_spins] *= -1;
        }
    }
}



void chimera_annealer::initialize(int num_samples, int num_sweeps, int sweeps_per_beta, double beta_start, double beta_end, bool use_geometric_schedule)
{
    this->num_samples = num_samples;

    if (states) delete[] states;
    if (energies) delete[] energies;
    if (delta_energies) delete[] delta_energies;

    states = new spin_t[num_spins * num_samples];
    energies = new double[num_samples];
    delta_energies = new float[num_spins * num_samples];
    random_state(states, num_samples * num_spins);
    
    if (rand_seeds) delete rand_seeds;
    rand_seeds = new uint32_t[num_samples];
    for (int i = 0; i < num_samples; i++)
        rand_seeds[i] = rng();
    
    double d_beta;
    num_beta_steps = num_sweeps / sweeps_per_beta;
    if (use_geometric_schedule && beta_start < 0.01) beta_start = 0.01;
    if (use_geometric_schedule)
        d_beta = pow(beta_end/beta_start, 1 / (double)num_beta_steps);
    else
        d_beta = (beta_end - beta_start) / num_beta_steps;
    
    if (betas) delete betas;
    betas = new float[1 + num_beta_steps];
    betas[0] = beta_start;
    for (int step = 1; step <= num_beta_steps; step++) {
        betas[step] = (use_geometric_schedule ? betas[step - 1] * d_beta : betas[step - 1] + d_beta);
    }
    
}

void chimera_annealer::calculate_delta_energies(int sample)
{
    for (int spin = 0; spin < num_spins; spin++) {
        delta_energies[spin + sample * num_spins] = fields[spin];
        for (int nbr_idx=0; nbr_idx<degrees[spin]; ++nbr_idx){
            int nbr = neighbours[nbr_idx + MAX_DEGREE * spin];
            float nbr_coupling = neighbour_couplings[nbr_idx + MAX_DEGREE * spin];
            delta_energies[spin + sample * num_spins] += nbr_coupling * states[nbr + sample * num_spins];
        }
        delta_energies[spin + sample * num_spins] *= (-2 * states[spin + sample * num_spins]);
    }
}



void chimera_annealer::calculate_energies()
{
    for (int k=0; k<num_samples; ++k) energies[k] = 0; // This line should be redundant.
    for (int i=0; i<num_spins; ++i){
        // Calculate energies due to fields.
        for (int sample=0; sample<num_samples; ++sample){
            energies[sample] += 2 * fields[i] * states[i + sample * num_spins];
        }
        // Calculate energies due to couplers.
        for (int j=0; j<degrees[i]; ++j){
            for (int k=0; k<num_samples; ++k){
                energies[k] += neighbour_couplings[j + MAX_DEGREE * i] * states[i + k * num_spins] * states[neighbours[j + i * MAX_DEGREE] + k * num_spins];
            }
        }
    }
    for (int k=0; k<num_samples; ++k) energies[k] /= 2;
}


