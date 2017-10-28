import unittest
import numpy as np
import copy

import dimod
from dwave_sage import DWaveSAGeSampler

class TestDimodWrapper(unittest.TestCase):
    def _get_simple_h_J(self):
        return {0: -1, 1: -1}, {(0, 1): -1}

    def _get_connected_FM_h_J(self, num_vars):
        h = {v: -1 for v in range(num_vars)}
        J = {(u, v): -1 for u in range(num_vars) for v in range(u, num_vars)}
        return h, J

    def _calc_entropy(self, response):
        samples = response.samples_array()
        _, counts = np.unique(samples, axis=0, return_counts=True)
        p = counts / float(len(samples))
        return -np.sum(p*np.log(p))

    def test_basic_response(self):
        sampler = DWaveSAGeSampler()
        h, J = self._get_simple_h_J()
        response = sampler.sample_ising(h, J)

        self.assertIsInstance(response, dimod.NumpySpinResponse,
                "Sampler returned an unexpected response type")

    def test_num_samples(self):
        sampler = DWaveSAGeSampler()
        h, J = self._get_simple_h_J()
        for num_samples in (1, 10, 100, 3223, 10352):
            response = sampler.sample_ising(h, J, num_samples=num_samples)
            self.assertEqual(len(list(response.samples())), num_samples,
                    "Sampler returned wrong number of samples")

        for bad_num_samples in (0, -1, -100):
            self.assertRaises(ValueError, sampler.sample_ising, h, J,
                    num_samples=bad_num_samples)

        for bad_num_samples in (3.5, float("inf"), "string", [], {}):
            self.assertRaises(TypeError, sampler.sample_ising, h, J,
                    num_samples=bad_num_samples)

    def test_empty_problem(self):
        sampler = DWaveSAGeSampler()
        nh, nJ = self._get_simple_h_J()
        eh, eJ = {}, {}
        
        for h in (nh, eh):
            for J in (nJ, eJ):
                _h = copy.deepcopy(h)
                _J = copy.deepcopy(J)
                r = sampler.sample_ising(_h, _J)


    def test_beta_range(self):
        """Tests at a very low beta, should be close to random results"""
        sampler = DWaveSAGeSampler()
        h, J = self._get_connected_FM_h_J(15)
        beta_range = (0.00001, 0.00002) # very low beta
        num_samples = 1000

        response = sampler.sample_ising(h, J, sweeps=10000, 
                num_samples=num_samples, beta_range=beta_range)
        entropy = self._calc_entropy(response)

        # with beta_range = (0.001, 4) (that is, a reasonable range for the
        # problem), average entropy over 100 seeds is ~ 0.4
        # with beta_range = (0.00001, 0.00002), so results should be very close
        # to completely random, average entropy over 100 seeds is ~ 6.9

        self.assertGreater(entropy, 5.0,
                msg=("Low entropy {} results on low beta problem, may be an "
                     "issue with the PRNG").format(entropy))

    def test_seed(self):
        sampler = DWaveSAGeSampler()
        num_vars = 40
        h, J = self._get_connected_FM_h_J(num_vars)
        num_samples = 1000

        # test seed exceptions
        for bad_seed in (3.5, float("inf"), "string", [], {}):
            self.assertRaises(TypeError, sampler.sample_ising, {}, {},
                    seed=bad_seed)
        for bad_seed in (-1, -100, 2**65):
            self.assertRaises(ValueError, sampler.sample_ising, {}, {},
                    seed=bad_seed)

        # make sure it can accept large seeds
        sampler.sample_ising(h, J, seed=2**63, num_samples=1, sweeps=1)

        # no need to do a bunch of sweeps, in fact the less we do the more
        # sure we can be that the same seed is returning the same result
        all_samples = []

        for seed in (1, 25, 2352, 736145, 5682453, 923759283623):
            response0 = sampler.sample_ising(h, J, num_samples=num_samples,
                    sweeps=10, seed=seed)
            response1 = sampler.sample_ising(h, J, num_samples=num_samples,
                    sweeps=10, seed=seed)

            samples0 = response0.samples_array()
            samples1 = response1.samples_array()

            self.assertTrue(np.array_equal(samples0, samples1),
                    "Same seed returned different results")

            for previous_sample in all_samples:
                self.assertFalse(np.array_equal(samples0, previous_sample),
                        "Different seed returned same results")

            all_samples.append(samples0)

    def test_disconnected_problem(self):
        sampler = DWaveSAGeSampler()
        h = {}
        J = {
                # K_3
                (0, 1): -1,
                (1, 2): -1,
                (0, 2): -1,

                # disonnected K_3
                (3, 4): -1,
                (4, 5): -1,
                (3, 5): -1,
                }

        response = sampler.sample_ising(h, J, sweeps=1000, num_samples=100)
        energies = list(response.energies())
        avg_energy = sum(energies) / float(len(energies))
        self.assertAlmostEqual(-6, avg_energy, delta=3,
                msg="Poor results ({}) on easy disconnected problem".format(avg_energy))


if __name__ == "__main__":
    unittest.main()
