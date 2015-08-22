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

#include "annealer.h"

using namespace std;


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
annealer::annealer(std::vector<double> field_vec, std::vector<int> coupler_starts, std::vector<int> coupler_ends, std::vector<double> coupler_values, bool rearrange, unsigned int seed)
{
    if (seed == 0) seed = std::chrono::high_resolution_clock::now().time_since_epoch().count(); // Default to time-based seed.
    this->seed = seed;
    rng = std::mt19937((uint32_t)seed);

    num_spins = field_vec.size();
    num_couplers = coupler_values.size();
    
    max_value = 1.0;
    
    this->fields = new float[num_spins]();

    if (rearrange) {
        for (unsigned i=0; i<field_vec.size(); i++) {
            fields[rearrange_vertex(i)] = field_vec[i];
            if (field_vec[i] > max_value) max_value = field_vec[i];
            else if (-field_vec[i] > max_value) max_value = -field_vec[i];
        }
        for (unsigned i=0; i<coupler_values.size(); i++) {
            coupler_starts[i] = rearrange_vertex(coupler_starts[i]);
            coupler_ends[i] = rearrange_vertex(coupler_ends[i]);
        }
    } else {
        for (unsigned i=0; i<field_vec.size(); i++)
            fields[i] = field_vec[i];
    }

    degrees = new int[num_spins]();
    for (int i=0; i<num_couplers; ++i) {
        degrees[coupler_starts[i]]++;
        degrees[coupler_ends[i]]++;
    }
    neighbours = new int[num_spins * MAX_DEGREE];
    neighbour_couplings = new float[num_spins * MAX_DEGREE];
    for (int i=0; i<num_spins; ++i) {
        degrees[i] = 0; // Reset so we can use them as counters
    }
    for (int i=0; i<num_couplers; ++i) {
        neighbours[degrees[coupler_starts[i]] + MAX_DEGREE * coupler_starts[i]] = coupler_ends[i];
        neighbours[degrees[coupler_ends[i]] + MAX_DEGREE * coupler_ends[i]] = coupler_starts[i];
        neighbour_couplings[degrees[coupler_starts[i]] + MAX_DEGREE * coupler_starts[i]] = coupler_values[i];
        neighbour_couplings[degrees[coupler_ends[i]] + MAX_DEGREE * coupler_ends[i]] = coupler_values[i];
        
        degrees[coupler_starts[i]]++;
        degrees[coupler_ends[i]]++;
        
        if (coupler_values[i] > max_value) max_value = coupler_values[i]; 
        else if (-coupler_values[i] > max_value) max_value = -coupler_values[i]; 
    }
    
    max_value__1 = 1.0 / max_value;
    
    for (int i = 0; i < EXP_TABSIZE * EXP_PRECISION; i++) {
        exptab[i] = exp(-((double)i * EXP_1_PRECISION));
    }

}

annealer::~annealer()
{
    delete[] fields;
    delete[] degrees;
    delete[] neighbours;
    delete[] neighbour_couplings;
    if (states) delete[] states;
    if (energies) delete[] energies;
    if (delta_energies) delete[] delta_energies;
    if (rand_seeds) delete[] rand_seeds;
    if (betas) delete[] betas;
}



void annealer::random_state(spin_t* destination, int num) {
    uint32_t rand = rng();
    for (int i=0; i<num; ++i) {
        if (i && !(i%32)) rand = rng();
        destination[i] = ((rand & 1) << 1) - 1;
        rand >>= 1;
    }
}


/** 
 * helper function to re-arrange vertex indices based on graph bipartition
 * 
 * @param v vertex index in standard chimera representation
 * @return index of v in bipartite representation
 */
inline int annealer::rearrange_vertex(int v)
{
    int n = sqrt(num_spins / 8);
    int x, y, r, s;
    
    r = v & 3;
    s = (v & 4) >> 2;
    y = (v >> 3) % n;
    x = (v >> 3) / n;
    
    int part = (x ^ y ^ s) & 1;
    return ((x * n * 4 + y * 4 + r) << 1) | part;
}


void annealer::calculate_energies()
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


/**
 * returns final energies of the solutions found by SA
 * 
 * @return final energies
 */
std::vector<double> annealer::get_energies() {
    std::vector<double> energy_vector = std::vector<double>();
    for (int i=0; i<num_samples; ++i) energy_vector.push_back((double)energies[i]);
    return energy_vector;
}

/**
 * returns final states of the solutions found by SA
 * 
 * @return final states
 */
std::vector< std::vector< int > > annealer::get_states() {
    std::vector< std::vector< int > > state_matrix = std::vector< std::vector< int > >();
    for (int i=0; i<num_samples; ++i) {
        std::vector<int> row = std::vector<int>();
        for (int j=0; j<num_spins; ++j){
            row.push_back((int)states[j + i * num_spins]);
        }
        state_matrix.push_back(row);
    }
    return state_matrix;
}



void annealer::print_states()
{
    std::cout << "states:" << std::endl;
    std::cout << "------------" << std::endl;
    for (int sample=0; sample<num_samples; ++sample) {
        for (int spin=0; spin<num_spins; ++spin) {
            std::cout << (int)states[spin + sample*num_spins] << ","; // Spin i of sample j is stored in states[j + i*num_samples]
        }
        std::cout << std::endl;
    }
    std::cout << std::endl << "------------" << std::endl;
}

void annealer::print_energies(bool verbose)
{
    if (verbose) std::cout << "energies:" << std::endl;
    if (verbose) std::cout << "------------" << std::endl;
    for (int i=0; i<num_samples; ++i) std::cout << std::setw(8) << energies[i];
    if (verbose) std::cout << std::endl << "------------" << std::endl;
    else std::cout << std::endl;
    
}

void annealer::print_energy_histogram()
{
    map<int, int> hist;
    for (int i=0; i<num_samples; i++)
        hist[energies[i]]++;
    
    for (auto it = hist.begin(); it != hist.end(); it++) {
        printf("%d\t%d\n", it->first, it->second);
    }
}

