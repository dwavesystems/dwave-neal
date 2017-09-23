import unittest
import numpy as np

import dwave_sage_sampler

class TestSA(unittest.TestCase):
    def _sample_fm_problem(self, num_variables=10, num_samples=100, 
                           num_sweeps=1000):
        h = [-1]*num_variables
        (coupler_starts, 
            coupler_ends, 
            coupler_weights) = zip(*((u,v,-1) for u in range(num_variables) 
                    for v in range(u, num_variables)))
        
        beta_schedule = np.linspace(0.01, 3, num=num_sweeps)
        seed = 1

        return (num_samples, h, coupler_starts, coupler_ends, coupler_weights,
                beta_schedule, seed)

    def test_submit_problem(self):
        num_variables, num_samples = 10, 100
        problem = self._sample_fm_problem(num_variables=num_variables,
                                          num_samples=num_samples)

        result = dwave_sage_sampler.simulated_annealing(*problem)

        self.assertTrue(len(result) == 2,
                "Sampler should return two values")

        samples, energies = result

        # ensure samples are all valid samples
        self.assertTrue(type(samples) is np.ndarray)
        # ensure correct number of samples and samples are have the correct
        # length
        self.assertTrue(samples.shape == (num_samples, num_variables), 
                "Sampler returned wrong shape for samples")
        # make sure samples contain only +-1
        self.assertTrue(set(np.unique(samples)) == {-1, 1},
                "Sampler returned spins with values not equal to +-1") 

        # ensure energies is valid
        self.assertTrue(type(energies) is np.ndarray)
        # one energy per sample
        self.assertTrue(energies.shape == (num_samples,),
                "Sampler returned wrong number of energies") 


    def test_good_results(self):
        num_variables = 5
        problem = self._sample_fm_problem(num_variables=num_variables)
        
        samples, energies = dwave_sage_sampler.simulated_annealing(*problem)

        ground_state = [1]*num_variables
        ground_energy = -(num_variables+3)*num_variables/2

        # we should definitely have gotten to the ground state
        self.assertTrue(ground_state in samples,
                "Ground state not found in samples from easy problem")

        mean_energy = np.mean(energies)
        self.assertAlmostEqual(ground_energy, mean_energy, delta=2,
                msg="Sampler returned poor mean energy for easy problem")

if __name__ == "__main__":
    unittest.main()
