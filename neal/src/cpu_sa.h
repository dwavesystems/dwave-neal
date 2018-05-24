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

#ifndef _cpu_sa_h
#define _cpu_sa_h

#ifdef _MSC_VER
// add uint64_t definition for windows
typedef __int64 int64_t;
typedef unsigned __int64 uint64_t;
#endif

double get_flip_energy(int var, char *state, std::vector<double> & h, 
                      std::vector<int> & degrees, 
                      std::vector<std::vector<int>> & neighbors, 
                      std::vector<std::vector<double>> & neighbour_couplings);

void simulated_annealing_run(char *state, std::vector<double> & h, 
                     std::vector<int> & degrees, 
                     std::vector<std::vector<int>> & neighbors, 
                     std::vector<std::vector<double>> & neighbour_couplings,
                     int sweeps_per_beta,
                     std::vector<double> beta_schedule);

double get_state_energy(char *state, std::vector<double> h, 
                        std::vector<int> coupler_starts, 
                        std::vector<int> coupler_ends, 
                        std::vector<double> coupler_values);

std::vector<double> general_simulated_annealing(char *states, 
                                          const int num_samples,
                                          std::vector<double> h, 
                                          std::vector<int> coupler_starts, 
                                          std::vector<int> coupler_ends, 
                                          std::vector<double> coupler_values,
                                          int sweeps_per_beta,
                                          std::vector<double> beta_schedule,
                                          uint64_t seed);

#endif
