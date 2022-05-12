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

import time
import unittest
import contextlib
from concurrent.futures import ThreadPoolExecutor, wait
from copy import deepcopy

import numpy as np

from neal.simulated_annealing import simulated_annealing


try:
    perf_counter = time.perf_counter
except AttributeError:  # pragma: no cover
    # python 2
    perf_counter = time.time


@contextlib.contextmanager
def tictoc(*args, **kwargs):
    """Code block execution timer.

    Example:

        with tictoc() as timer:
            # some code
            print("partial duration", timer.duration)
            # more code

        print("total duration", timer.duration)

    """

    class Timer(object):
        _start = _end = None

        def start(self):
            self._start = perf_counter()

        def stop(self):
            self._end = perf_counter()

        @property
        def duration(self):
            if self._start is None:
                return None
            if self._end is not None:
                return self._end - self._start
            return perf_counter() - self._start

    timer = Timer()
    timer.start()
    try:
        yield timer
    finally:
        timer.stop()


def cpu_count():
    try:
        import os
        # doesn't exist in python2, and can return None
        return os.cpu_count() or 1
    except AttributeError:
        pass

    try:
        import multiprocessing
        # doesn't have to be implemented
        return multiprocessing.cpu_count()
    except NotImplementedError:
        pass

    return 1


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

        np_rand = np.random.RandomState(1234)
        initial_states = 2*np_rand.randint(2, size=(num_samples, num_variables)).astype(np.int8) - 1

        return (num_samples, h, coupler_starts, coupler_ends, coupler_weights,
                sweeps_at_beta, beta_schedule, seed, initial_states)

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
        self.assertTrue(set(np.unique(samples)).issubset({-1, 1}),
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
        # problem = self._sample_fm_problem(num_variables=40, num_samples=1000, num_sweeps=10)
        num_variables, num_sweeps, num_samples = 100, 10, 1000
        h = [0]*num_variables
        (coupler_starts,
         coupler_ends,
         coupler_weights) = zip(*((u, v, 1) for u in range(num_variables)
                                  for v in range(u, num_variables)))

        beta_schedule = np.linspace(0.3, 0.4, num=num_sweeps)
        sweeps_at_beta = 1

        np_rand = np.random.RandomState(1234)
        initial_states = np_rand.randint(1, size=(num_samples, num_variables))
        initial_states = 2*initial_states.astype(np.int8) - 1

        previous_samples = []
        for seed in (1, 40, 235, 152436, 3462354, 92352355):
            samples0, _ = simulated_annealing(num_samples, h, coupler_starts,
                                              coupler_ends, coupler_weights,
                                              sweeps_at_beta, beta_schedule,
                                              seed, np.copy(initial_states))
            samples1, _ = simulated_annealing(num_samples, h, coupler_starts,
                                              coupler_ends, coupler_weights,
                                              sweeps_at_beta, beta_schedule,
                                              seed, np.copy(initial_states))

            self.assertTrue(np.array_equal(samples0, samples1),
                            "Same seed returned different results")

            for previous_sample in previous_samples:
                self.assertFalse(np.array_equal(samples0, previous_sample),
                                 "Different seed returned same results")

            previous_samples.append(samples0)

    def test_immediate_interrupt(self):
        num_variables = 5
        problem = self._sample_fm_problem(num_variables=num_variables)

        # should only get one sample back
        samples, energies = simulated_annealing(*problem, interrupt_function=lambda: True)

        self.assertEqual(samples.shape, (1, 5))
        self.assertEqual(energies.shape, (1,))

    def test_5_interrupt(self):
        num_variables = 5
        problem = self._sample_fm_problem(num_variables=num_variables)

        count = [1]

        def stop():
            if count[0] >= 5:
                return True
            count[0] += 1
            return False

        # should only get one sample back
        samples, energies = simulated_annealing(*problem, interrupt_function=stop)

        self.assertEqual(samples.shape, (5, 5))
        self.assertEqual(energies.shape, (5,))

    def test_initial_states(self):
        num_variables, num_sweeps, num_samples = 100, 0, 1000
        h = [0]*num_variables
        (coupler_starts,
         coupler_ends,
         coupler_weights) = zip(*((u, v, 1) for u in range(num_variables)
                                  for v in range(u, num_variables)))

        beta_schedule = np.linspace(0.3, 0.4, num=num_sweeps)
        sweeps_at_beta = 1
        seed = 1234567890

        np_rand = np.random.RandomState(1234)
        initial_states = np_rand.randint(1, size=(num_samples, num_variables))
        initial_states = 2*initial_states.astype(np.int8) - 1

        samples, _ = simulated_annealing(num_samples, h, coupler_starts,
                                         coupler_ends, coupler_weights,
                                         sweeps_at_beta, beta_schedule,
                                         seed, np.copy(initial_states))

        self.assertTrue(np.array_equal(initial_states, samples),
                        "Initial states do not match samples with 0 sweeps")

    @unittest.skipUnless(cpu_count() >= 2, "at least two threads required")
    def test_concurrency(self):
        """Multiple SA run in parallel threads, not blocking each other due to GIL."""

        problem = self._sample_fm_problem(
            num_variables=100, num_samples=100, num_sweeps=10000)

        num_threads = 2

        with ThreadPoolExecutor(max_workers=num_threads) as executor:

            with tictoc() as sequential:
                for _ in range(num_threads):
                    wait([executor.submit(simulated_annealing, *deepcopy(problem))])

            with tictoc() as parallel:
                wait([executor.submit(simulated_annealing, *deepcopy(problem))
                    for _ in range(num_threads)])

        speedup = sequential.duration / parallel.duration

        # NOTE: we would like to assert stricter bounds on the speedup, e.g.:
        #   self.assertGreater(speedup, 0.75*num_threads)
        #   self.assertLess(speedup, 1.25*num_threads)
        # but due to unreliable/inconsistent performance on CI VMs, we have
        # to settle with a very basic constraint of >0% speedup, which
        # indicates, at least, some minimal level of parallelism
        self.assertGreater(speedup, 1)


if __name__ == "__main__":
    unittest.main()
