# Copyright 2018 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# ================================================================================================

import unittest
import numpy as np

from neal.simulated_annealing import simulated_annealing


class TestSA(unittest.TestCase):
    def _sample_fm_problem(self, num_variables=10, num_samples=100, num_sweeps=1000):
        h = [-1]*num_variables
        (coupler_starts,
            coupler_ends,
            coupler_weights) = zip(*((u, v, -1) for u in range(num_variables)
                                   for v in range(u, num_variables)))

        beta_schedule = np.linspace(0.01, 3, num=num_sweeps)
        sweeps_at_beta = 1
        seed = 1

        return (num_samples, h, coupler_starts, coupler_ends, coupler_weights,
                sweeps_at_beta, beta_schedule, seed)

    def test_submit_problem(self):
        num_variables, num_samples = 10, 100
        problem = self._sample_fm_problem(num_variables=num_variables,
                                          num_samples=num_samples)

        result = simulated_annealing(*problem)

        self.assertTrue(len(result) == 2, "Sampler should return two values")

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

        samples, energies = simulated_annealing(*problem)

        ground_state = [1]*num_variables
        ground_energy = -(num_variables+3)*num_variables/2

        # we should definitely have gotten to the ground state
        self.assertTrue(ground_state in samples,
                        "Ground state not found in samples from easy problem")

        mean_energy = np.mean(energies)
        self.assertAlmostEqual(ground_energy, mean_energy, delta=2,
                               msg="Sampler returned poor mean energy for easy problem")

    def test_seed(self):
        # no need to do a bunch of sweeps, in fact the less we do the more
        # sure we can be that the same seed is returning the same result
        problem = self._sample_fm_problem(num_variables=40, num_samples=1000, num_sweeps=10)

        # no need to do a bunch of sweeps, in fact the less we do the more
        # sure we can be that the same seed is returning the same result

        previous_samples = []
        for seed in (1, 40, 235, 152436, 3462354, 92352355):
            seeded_problem = problem[:-1] + (seed,)
            samples0, _ = simulated_annealing(*seeded_problem)
            samples1, _ = simulated_annealing(*seeded_problem)

            self.assertTrue(np.array_equal(samples0, samples1), "Same seed returned different results")

            for previous_sample in previous_samples:
                self.assertFalse(np.array_equal(samples0, previous_sample), "Different seed returned same results")

            previous_samples.append(samples0)


if __name__ == "__main__":
    unittest.main()
