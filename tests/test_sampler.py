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
import warnings
import collections

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

    def test_answer_mode(self):
        sampler = Neal()
        h = {'a': 0, 'b': -1, 'c': 2}
        J = {('a', 'b'): -1, ('b', 'c'): 2, ('a', 'c'): 4}
        response1 = sampler.sample_ising(h, J, answer_mode='histogram', num_reads=1)
        response2 = sampler.sample_ising(h, J, answer_mode='raw', num_reads=1)
        self.assertEqual(response1, response2)

        h = {i:i-15 for i in range(30)}
        J = {(i, j) : i+j-20 for i in range(30) for j in range(30) if i < j}

        num_reads = 100
        response1 = sampler.sample_ising(h, J, answer_mode='histogram', num_reads=num_reads)
        response2 = sampler.sample_ising(h, J, answer_mode='raw', num_reads=num_reads)
        self.assertEqual(num_reads, response1.record.num_occurrences.sum())
        self.assertEqual(num_reads, response2.record.num_occurrences.sum())

        energy_prev = -np.inf
        for sample, energy, num_occurrences in response1.data():
            bqm = dimod.BinaryQuadraticModel(h, J, offset=0, vartype=dimod.SPIN)
            self.assertEqual(bqm.energy(sample), energy)
            self.assertGreaterEqual(energy, energy_prev)
            energy_prev=energy

        count_dict1 = collections.defaultdict(int)
        count_dict2 = collections.defaultdict(int)
        for sample, energy, num_occurrences in response1.data():
            count_dict1[tuple(sample.values())] += num_occurrences
        for sample, energy, num_occurrences in response2.data():
            count_dict2[tuple(sample.values())] += num_occurrences

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
        sampler.sample_ising(h, J, seed=2**63, num_reads=1, num_sweeps=1)

        # no need to do a bunch of sweeps, in fact the less we do the more
        # sure we can be that the same seed is returning the same result
        all_samples = []

        for seed in (1, 25, 2352, 736145, 5682453, 923759283623):
            response0 = sampler.sample_ising(h, J, num_reads=num_reads, num_sweeps=10, seed=seed)
            response1 = sampler.sample_ising(h, J, num_reads=num_reads, num_sweeps=10, seed=seed)

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

        resp = sampler.sample_ising(h, J, num_sweeps=1000, num_reads=100)

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

    def test_ising_tuple_initial_states(self):
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

        # sample, but ignore deprecation warnings (tuple format for initial_states)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resp = sampler.sample_ising(h, J, num_reads=num_reads, num_sweeps=0, seed=seed,
                                        initial_states=(initial_state_array, init_labels))

        for v in var_labels:
            self.assertTrue(np.array_equal(resp.record.sample[:, resp.variables.index(v)], 
                                           initial_state_array[:, init_labels[v]]),
                            "Samples were not the same as initial states with "
                            "no sweeps")

    def test_binary_tuple_initial_states(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({}, {(0,1): 1}).binary
        num_vars = len(bqm)
        num_reads = 10

        initial_states_array = np.random.randint(2, size=(num_reads, num_vars))

        expected_response = dimod.SampleSet.from_samples_bqm(initial_states_array, bqm)

        # sample, but ignore deprecation warnings (tuple format for initial_states)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            response = Neal().sample(bqm, num_sweeps=0, initial_states=(initial_states_array, None))

        # if initial_states are not properly converted by the sample method
        # (from binary to ising), energy levels will not match
        self.assertTrue(np.array_equal(response.record.sample, expected_response.record.sample))
        self.assertTrue(np.array_equal(response.record.energy, expected_response.record.energy))

    def test_sampleset_initial_states(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({}, {'ab': 1, 'bc': 1, 'ca': 1})
        initial_states = dimod.SampleSet.from_samples_bqm({'a': 1, 'b': -1, 'c': 1}, bqm)

        response = Neal().sample(bqm, initial_states=initial_states, num_reads=1)

        self.assertEqual(len(response), 1)
        self.assertEqual(response.first.energy, -1)

    def test_initial_states_generator(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({}, {'ab': -1, 'bc': 1, 'ac': 1})
        init = dimod.SampleSet.from_samples_bqm([{'a': 1, 'b': 1, 'c': 1},
                                                 {'a': -1, 'b': -1, 'c': -1}], bqm)
        sampler = Neal()

        # 2 fixed initial state, 8 random
        resp = sampler.sample(bqm, initial_states=init, num_reads=10)
        self.assertEqual(len(resp), 10)

        # 2 fixed initial states, 8 random, explicit
        resp = sampler.sample(bqm, initial_states=init, initial_states_generator='random', num_reads=10)
        self.assertEqual(len(resp), 10)

        # all random
        resp = sampler.sample(bqm, initial_states_generator='random', num_reads=10)
        self.assertEqual(len(resp), 10)

        # all random
        resp = sampler.sample(bqm, num_reads=10)
        self.assertEqual(len(resp), 10)


        # zero-length init states in tuple format, extended by random samples
        zero_init_tuple = (np.empty((0, 3)), None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resp = sampler.sample(bqm, initial_states=zero_init_tuple, num_reads=10)
        self.assertEqual(len(resp), 10)

        # explicit None for initial_states should use one random init state
        resp = sampler.sample(bqm, initial_states=None)
        self.assertEqual(len(resp), 1)

        # initial_states truncated to num_reads?
        resp = sampler.sample(bqm, initial_states=init, initial_states_generator='none', num_reads=1)
        self.assertEqual(len(resp), 1)

        resp = sampler.sample(bqm, initial_states=init, initial_states_generator='tile', num_reads=1)
        self.assertEqual(len(resp), 1)

        resp = sampler.sample(bqm, initial_states=init, initial_states_generator='random', num_reads=1)
        self.assertEqual(len(resp), 1)


        # 2 fixed initial states, repeated 5 times
        resp = sampler.sample(bqm, initial_states=init, initial_states_generator='tile', num_reads=10)
        self.assertEqual(len(resp), 10)

        # can't tile empty states
        with self.assertRaises(ValueError):
            resp = sampler.sample(bqm, initial_states_generator='tile', num_reads=10)

        # not enough initial states
        with self.assertRaises(ValueError):
            resp = sampler.sample(bqm, initial_states_generator='none', num_reads=3)

        # initial_states incompatible with the bqm
        init = dimod.SampleSet.from_samples({'a': 1, 'b': 1}, vartype='SPIN', energy=0)
        with self.assertRaises(ValueError):
            resp = sampler.sample(bqm, initial_states=init)

    def test_soft_num_reads(self):
        """Number of reads adapts to initial_states size, if provided."""

        bqm = dimod.BinaryQuadraticModel.from_ising({}, {'ab': -1, 'bc': 1, 'ac': 1})
        init = dimod.SampleSet.from_samples_bqm([{'a': 1, 'b': 1, 'c': 1},
                                                 {'a': -1, 'b': -1, 'c': -1}], bqm)
        sampler = Neal()

        # default num_reads == 1
        self.assertEqual(len(sampler.sample(bqm)), 1)
        self.assertEqual(len(sampler.sample(bqm, initial_states_generator="random")), 1)

        # with initial_states, num_reads == len(initial_states)
        self.assertEqual(len(sampler.sample(bqm, initial_states=init)), 2)

        # ... but explicit truncation works too
        self.assertEqual(len(sampler.sample(bqm, initial_states=init, num_reads=1)), 1)

        # if num_reads explicitly given together with initial_states, they are expanded
        self.assertEqual(len(sampler.sample(bqm, initial_states=init, num_reads=3)), 3)

        # if num_reads explicitly given together without initial_states, they are generated
        self.assertEqual(len(sampler.sample(bqm, num_reads=4)), 4)

    def test_deprecated_sweeps_name(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({}, {'ab': 1})
        sampler = Neal()

        with warnings.catch_warnings():
            warnings.simplefilter("error")

            with self.assertRaises(DeprecationWarning):
                sampler.sample(bqm, sweeps=10)


class TestDefaultBetaRange(unittest.TestCase):
    def test_empty_problem(self):
        self.assertEqual(neal.sampler._default_ising_beta_range({}, {}), [.1, 1.0])

    def test_single_variable_ising_problem(self):
        h1, c1 = neal.sampler._default_ising_beta_range({'a': 0.1}, {})
        h2, c2 = neal.sampler._default_ising_beta_range({'a': 1}, {})
        h3, c3 = neal.sampler._default_ising_beta_range({'a': 10}, {})

        self.assertTrue(h1 > h2 > h3)
        self.assertTrue(c1 > c2 > c3)
        self.assertTrue(h1 < c1 and h2 < c2 and h3 < c3)

    def test_single_coupling_ising_problem(self):
        h1, c1 = neal.sampler._default_ising_beta_range({}, {'ab': 0.1})
        h2, c2 = neal.sampler._default_ising_beta_range({}, {'ab': 1})
        h3, c3 = neal.sampler._default_ising_beta_range({}, {'ab': 10})

        self.assertTrue(h1 > h2 > h3)
        self.assertTrue(c1 > c2 > c3)
        self.assertTrue(h1 < c1 and h2 < c2 and h3 < c3)

    def test_bias_coupling_ranges(self):
        h1, c1 = neal.sampler._default_ising_beta_range({'a': 1}, {'ab': 1})
        h2, c2 = neal.sampler._default_ising_beta_range({'a': 10}, {'ab': 1})
        h3, c3 = neal.sampler._default_ising_beta_range({'a': 10}, {'ab': 10})

        self.assertTrue(h1 > h2 > h3)
        self.assertTrue(c1 == c2 > c3)
        self.assertTrue(h1 < c1 and h2 < c2 and h3 < c3)

    def test_default_beta_range(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({'a': 1}, {'bc': 1})
        self.assertEqual(neal.default_beta_range(bqm),
                         neal.default_beta_range(bqm.binary))


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
        response = sampler.sample(jss_bqm, beta_schedule_type="linear", num_reads=10)
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
        response = sampler.sample_ising({}, J, beta_schedule_type="geometric", num_reads=10)
        _, response_energy, _ = next(response.data())

        # Note: lowest energy found was -3088 with a different benchmarking tool
        threshold = -3000
        self.assertLess(response_energy, threshold, ("response_energy, {}, exceeds "
            "threshold").format(response_energy))


if __name__ == "__main__":
    unittest.main()
