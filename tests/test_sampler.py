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
import itertools

import dimod

import neal
from neal import Neal


class TestSimulatedAnnealingSampler(unittest.TestCase):
    def test_instantiation(self):
        sampler = Neal()
        dimod.testing.assert_sampler_api(sampler)

    def test_one_node_beta_range(self):
        h = {'a': -1}
        bqm = dimod.BinaryQuadraticModel(h, {}, 0, dimod.SPIN)
        response = Neal().sample(bqm)
        hot_beta, cold_beta = response.info['beta_range']

        # Check beta values
        # Note: beta is proportional to 1/temperature, therefore hot_beta < cold_beta
        self.assertLess(hot_beta, cold_beta)
        self.assertNotEqual(hot_beta, float("inf"), "Starting value of 'beta_range' is infinite")
        self.assertNotEqual(cold_beta, float("inf"), "Final value of 'beta_range' is infinite")

    def test_one_edge_beta_range(self):
        J = {('a', 'b'): 1}
        bqm = dimod.BinaryQuadraticModel({}, J, 0, dimod.BINARY)
        response = Neal().sample(bqm)
        hot_beta, cold_beta = response.info['beta_range']

        # Check beta values
        # Note: beta is proportional to 1/temperature, therefore hot_beta < cold_beta
        self.assertLess(hot_beta, cold_beta)
        self.assertNotEqual(hot_beta, float("inf"), "Starting value of 'beta_range' is infinite")
        self.assertNotEqual(cold_beta, float("inf"), "Final value of 'beta_range' is infinite")

    def test_sample_ising(self):
        h = {'a': 0, 'b': -1}
        J = {('a', 'b'): -1}

        resp = Neal().sample_ising(h, J)

        row, col = resp.record.sample.shape

        self.assertEqual(col, 2)  # should get back two variables
        self.assertIs(resp.vartype, dimod.SPIN)  # should be ising

    def test_sample_qubo(self):
        Q = {(0, 1): 1}
        resp = Neal().sample_qubo(Q)

        row, col = resp.record.sample.shape

        self.assertEqual(col, 2)  # should get back two variables
        self.assertIs(resp.vartype, dimod.BINARY)  # should be qubo

    def test_basic_response(self):
        sampler = Neal()
        h = {'a': 0, 'b': -1}
        J = {('a', 'b'): -1}
        response = sampler.sample_ising(h, J)

        self.assertIsInstance(response, dimod.SampleSet, "Sampler returned an unexpected response type")

    def test_num_reads(self):
        sampler = Neal()

        h = {}
        J = {('a', 'b'): .5, (0, 'a'): -1, (1, 'b'): 0.0}

        for num_reads in (1, 10, 100, 3223, 10352):
            response = sampler.sample_ising(h, J, num_reads=num_reads)
            row, col = response.record.sample.shape

            self.assertEqual(row, num_reads)
            self.assertEqual(col, 4)

        for bad_num_reads in (0, -1, -100):
            with self.assertRaises(ValueError):
                sampler.sample_ising(h, J, num_reads=bad_num_reads)

        for bad_num_reads in (3.5, float("inf"), "string", [], {}):
            with self.assertRaises(TypeError):
                sampler.sample_ising(h, J, num_reads=bad_num_reads)

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
        num_reads = 1000

        # test seed exceptions
        for bad_seed in (3.5, float("inf"), "string", [], {}):
            self.assertRaises(TypeError, sampler.sample_ising, {}, {}, seed=bad_seed)
        for bad_seed in (-1, -100, 2**65):
            self.assertRaises(ValueError, sampler.sample_ising, {}, {}, seed=bad_seed)

        # make sure it can accept large seeds
        sampler.sample_ising(h, J, seed=2**63, num_reads=1, sweeps=1)

        # no need to do a bunch of sweeps, in fact the less we do the more
        # sure we can be that the same seed is returning the same result
        all_samples = []

        for seed in (1, 25, 2352, 736145, 5682453, 923759283623):
            response0 = sampler.sample_ising(h, J, num_reads=num_reads, sweeps=10, seed=seed)
            response1 = sampler.sample_ising(h, J, num_reads=num_reads, sweeps=10, seed=seed)

            samples0 = response0.record.sample
            samples1 = response1.record.sample

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

        resp = sampler.sample_ising(h, J, sweeps=1000, num_reads=100)

        row, col = resp.record.sample.shape

        self.assertEqual(row, 100)
        self.assertEqual(col, 6)  # should get back two variables
        self.assertIs(resp.vartype, dimod.SPIN)  # should be ising

    def test_geometric_schedule(self):
        sampler = Neal()
        num_vars = 40
        h = {v: -1 for v in range(num_vars)}
        J = {(u, v): -1 for u in range(num_vars) for v in range(u, num_vars) if u != v}
        num_reads = 10

        resp = sampler.sample_ising(h, J, num_reads=num_reads, beta_schedule_type='geometric')

        row, col = resp.record.sample.shape

        self.assertEqual(row, num_reads)
        self.assertEqual(col, num_vars)  # should get back two variables
        self.assertIs(resp.vartype, dimod.SPIN)  # should be ising

        with self.assertRaises(ValueError):
            sampler.sample_ising(h, J, num_reads=num_reads, beta_schedule_type='asd')

    def test_interrupt_error(self):
        sampler = Neal()
        num_vars = 40
        h = {v: -1 for v in range(num_vars)}
        J = {(u, v): -1 for u in range(num_vars) for v in range(u, num_vars) if u != v}
        num_reads = 100

        def f():
            raise NotImplementedError

        resp = sampler.sample_ising(h, J, num_reads=num_reads, interrupt_function=f)

        self.assertEqual(len(resp), 1)

    def test_initial_states(self):
        sampler = Neal()
        var_labels = ["a", "b", "c", "d"]
        num_vars = len(var_labels)
        h = {v: -1 for v in var_labels}
        J = {(u, v): 1 for u, v in itertools.combinations(var_labels, 2)}
        num_reads = 100
        seed = 1234567890

        np_rand = np.random.RandomState(seed)
        initial_state_array = 2*np_rand.randint(2, size=(num_reads, num_vars)) - 1
        init_labels = dict(zip(var_labels, np_rand.permutation(num_vars)))

        resp = sampler.sample_ising(h, J, num_reads=num_reads, sweeps=0, seed=seed, 
                                    initial_states=(initial_state_array, init_labels))

        for v in var_labels:
            self.assertTrue(np.array_equal(resp.record.sample[:, resp.variables.index(v)], 
                                           initial_state_array[:, init_labels[v]]),
                            "Samples were not the same as initial states with "
                            "no sweeps")


class TestDefaultIsingBetaRange(unittest.TestCase):
    def test_empty_problem(self):
        self.assertEqual(neal.sampler._default_ising_beta_range({}, {}), [.1, 1.0])


class TestHeuristicResponse(unittest.TestCase):
    def test_job_shop_scheduling_with_linear(self):
        # Set up a job shop scheduling BQM
        #
        # Provide hardcode version of the bqm of "jobs"
        #   jobs = {'b': [(1,1), (3,1)],
        #           'o': [(2,2), (4,1)],
        #           'g': [(1,2)]}
        #
        #   There are three jobs: 'b', 'o', 'g'
        #   Each tuple represents a task that runs on a particular machine for a given amount of
        #   time. I.e. (machine_id, duration_on_machine)
        #
        #   Variables below are labelled as '<job_name>_<task_index>,<task_start_time>'.
        linear = {'b_0,0': -2.0,
                  'b_0,1': -2.0,
                  'b_0,2': -2.0,
                  'b_0,3': -2.0,
                  'b_1,0': 0.125,
                  'b_1,1': -1.5,
                  'b_1,2': 0.0,
                  'g_0,0': -1.875,
                  'g_0,1': -1.5,
                  'g_0,2': 0.0,
                  'o_0,0': -2.0,
                  'o_0,1': -2.0,
                  'o_0,2': -2.0,
                  'o_1,0': 0.03125,
                  'o_1,1': -1.875,
                  'o_1,2': -1.5,
                  'o_1,3': 0.0}

        quadratic = {('b_0,0', 'g_0,0'): 4,
                     ('b_0,1', 'b_0,0'): 4.0,
                     ('b_0,1', 'g_0,0'): 2,
                     ('b_0,2', 'b_0,0'): 4.0,
                     ('b_0,2', 'b_0,1'): 4.0,
                     ('b_0,2', 'b_1,2'): 2,
                     ('b_0,2', 'g_0,1'): 2,
                     ('b_0,2', 'g_0,2'): 4,
                     ('b_0,3', 'b_0,0'): 4.0,
                     ('b_0,3', 'b_0,1'): 4.0,
                     ('b_0,3', 'b_0,2'): 4.0,
                     ('b_0,3', 'b_1,2'): 2,
                     ('b_0,3', 'g_0,2'): 2,
                     ('b_1,1', 'b_0,1'): 2,
                     ('b_1,1', 'b_0,2'): 2,
                     ('b_1,1', 'b_0,3'): 2,
                     ('b_1,1', 'b_1,2'): 4.0,
                     ('g_0,1', 'b_0,1'): 4,
                     ('g_0,1', 'g_0,0'): 4.0,
                     ('g_0,2', 'g_0,0'): 4.0,
                     ('g_0,2', 'g_0,1'): 4.0,
                     ('o_0,0', 'o_1,1'): 2,
                     ('o_0,1', 'o_0,0'): 4.0,
                     ('o_0,1', 'o_1,1'): 2,
                     ('o_0,1', 'o_1,2'): 2,
                     ('o_0,2', 'o_0,0'): 4.0,
                     ('o_0,2', 'o_0,1'): 4.0,
                     ('o_0,2', 'o_1,1'): 2,
                     ('o_1,2', 'o_0,2'): 2,
                     ('o_1,2', 'o_1,1'): 4.0,
                     ('o_1,3', 'o_0,2'): 2,
                     ('o_1,3', 'o_1,1'): 4.0,
                     ('o_1,3', 'o_1,2'): 4.0}

        jss_bqm = dimod.BinaryQuadraticModel(linear, quadratic, 9.0, dimod.BINARY)

        # Optimal energy
        optimal_solution = {'b_0,0': 1, 'b_0,1': 0, 'b_0,2': 0, 'b_0,3': 0,
                            'b_1,0': 0, 'b_1,1': 1, 'b_1,2': 0, 'b_1,3': 0,
                            'g_0,0': 0, 'g_0,1': 1, 'g_0,2': 0, 'g_0,3': 0,
                            'o_0,0': 1, 'o_0,1': 0, 'o_0,2': 0, 'o_0,3': 0,
                            'o_1,0': 0, 'o_1,1': 0, 'o_1,2': 1, 'o_1,3': 0}
        optimal_energy = jss_bqm.energy(optimal_solution) # Evaluates to 0.5

        # Get heuristic solution
        sampler = Neal()
        response = sampler.sample(jss_bqm, beta_schedule_type="linear")
        _, response_energy, _ = next(response.data())

        # Compare energies
        threshold = 0.1	 # Arbitrary threshold
        self.assertLess(response_energy, optimal_energy + threshold)

    def test_cubic_lattice_with_geometric(self):
        # Set up all lattice edges in a cube. Each edge is labelled by a 3-D coordinate system
        def get_cubic_lattice_edges(N):
            for x, y, z in itertools.product(range(N), repeat=3):
                u = x, y, z
                yield u, ((x+1)%N, y, z)
                yield u, (x, (y+1)%N, z)
                yield u, (x, y, (z+1)%N)

        # Add a J-bias to each edge
        np_rand = np.random.RandomState(128)
        J = {e: np_rand.choice((-1, 1)) for e in get_cubic_lattice_edges(12)}

        # Solve ising problem
        sampler = Neal()
        response = sampler.sample_ising({}, J, beta_schedule_type="geometric")
        _, response_energy, _ = next(response.data())

        # Note: lowest energy found was -3088 with a different benchmarking tool
        threshold = -3000
        self.assertLess(response_energy, threshold, ("response_energy, {}, exceeds "
            "threshold").format(response_energy))


if __name__ == "__main__":
    unittest.main()
