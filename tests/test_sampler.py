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
import copy

import dimod

import neal
from neal import Neal


class TestSimulatedAnnealingSampler(unittest.TestCase):
    def test_instantiation(self):
        sampler = Neal()
        dimod.testing.assert_sampler_api(sampler)

    def test_sample_ising(self):
        h = {'a': 0, 'b': -1}
        J = {('a', 'b'): -1}

        resp = Neal().sample_ising(h, J)

        row, col = resp.samples_matrix.shape

        self.assertEqual(col, 2)  # should get back two variables
        self.assertIs(resp.vartype, dimod.SPIN)  # should be ising

    def test_sample_qubo(self):
        Q = {(0, 1): 1}
        resp = Neal().sample_qubo(Q)

        row, col = resp.samples_matrix.shape

        self.assertEqual(col, 2)  # should get back two variables
        self.assertIs(resp.vartype, dimod.BINARY)  # should be qubo

    def test_basic_response(self):
        sampler = Neal()
        h = {'a': 0, 'b': -1}
        J = {('a', 'b'): -1}
        response = sampler.sample_ising(h, J)

        self.assertIsInstance(response, dimod.Response, "Sampler returned an unexpected response type")

    def test_num_samples(self):
        sampler = Neal()

        h = {}
        J = {('a', 'b'): .5, (0, 'a'): -1, (1, 'b'): 0.0}

        for num_samples in (1, 10, 100, 3223, 10352):
            response = sampler.sample_ising(h, J, num_samples=num_samples)
            row, col = response.samples_matrix.shape

            self.assertEqual(row, num_samples)
            self.assertEqual(col, 4)

        for bad_num_samples in (0, -1, -100):
            with self.assertRaises(ValueError):
                sampler.sample_ising(h, J, num_samples=bad_num_samples)

        for bad_num_samples in (3.5, float("inf"), "string", [], {}):
            with self.assertRaises(TypeError):
                sampler.sample_ising(h, J, num_samples=bad_num_samples)

    def test_empty_problem(self):
        sampler = Neal()
        h = {'a': 0, 'b': -1}
        J = {('a', 'b'): -1}
        eh, eJ = {}, {}

        for h in (h, eh):
            for J in (J, eJ):
                _h = copy.deepcopy(h)
                _J = copy.deepcopy(J)
                r = sampler.sample_ising(_h, _J)

    def test_seed(self):
        sampler = Neal()
        num_vars = 40
        h = {v: -1 for v in range(num_vars)}
        J = {(u, v): -1 for u in range(num_vars) for v in range(u, num_vars) if u != v}
        num_samples = 1000

        # test seed exceptions
        for bad_seed in (3.5, float("inf"), "string", [], {}):
            self.assertRaises(TypeError, sampler.sample_ising, {}, {}, seed=bad_seed)
        for bad_seed in (-1, -100, 2**65):
            self.assertRaises(ValueError, sampler.sample_ising, {}, {}, seed=bad_seed)

        # make sure it can accept large seeds
        sampler.sample_ising(h, J, seed=2**63, num_samples=1, sweeps=1)

        # no need to do a bunch of sweeps, in fact the less we do the more
        # sure we can be that the same seed is returning the same result
        all_samples = []

        for seed in (1, 25, 2352, 736145, 5682453, 923759283623):
            response0 = sampler.sample_ising(h, J, num_samples=num_samples, sweeps=10, seed=seed)
            response1 = sampler.sample_ising(h, J, num_samples=num_samples, sweeps=10, seed=seed)

            samples0 = response0.samples_matrix
            samples1 = response1.samples_matrix

            self.assertTrue(np.array_equal(samples0, samples1), "Same seed returned different results")

            for previous_sample in all_samples:
                self.assertFalse(np.array_equal(samples0, previous_sample), "Different seed returned same results")

            all_samples.append(samples0)

    def test_disconnected_problem(self):
        sampler = Neal()
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

        resp = sampler.sample_ising(h, J, sweeps=1000, num_samples=100)

        row, col = resp.samples_matrix.shape

        self.assertEqual(row, 100)
        self.assertEqual(col, 6)  # should get back two variables
        self.assertIs(resp.vartype, dimod.SPIN)  # should be ising

    def test_geometric_schedule(self):
        sampler = Neal()
        num_vars = 40
        h = {v: -1 for v in range(num_vars)}
        J = {(u, v): -1 for u in range(num_vars) for v in range(u, num_vars) if u != v}
        num_samples = 10

        resp = sampler.sample_ising(h, J, num_samples=num_samples, beta_schedule_type='geometric')

        row, col = resp.samples_matrix.shape

        self.assertEqual(row, num_samples)
        self.assertEqual(col, num_vars)  # should get back two variables
        self.assertIs(resp.vartype, dimod.SPIN)  # should be ising

        with self.assertRaises(ValueError):
            sampler.sample_ising(h, J, num_samples=num_samples, beta_schedule_type='asd')


class TestDefaultIsingBetaRange(unittest.TestCase):
    def test_empty_problem(self):
        self.assertEqual(neal.sampler._default_ising_beta_range({}, {}), [.1, 1.0])


if __name__ == "__main__":
    unittest.main()
