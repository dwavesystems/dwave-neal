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
