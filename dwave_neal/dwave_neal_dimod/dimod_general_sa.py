from __future__ import division
import sys
from random import randint
import math

from dimod import TemplateSampler, NumpySpinResponse
from dimod.decorators import ising

from dwave_neal_sampler import simulated_annealing

if sys.version_info[0] == 2:
    range = xrange
    itervalues = lambda d: d.itervalues()
    iteritems = lambda d: d.iteritems()
    int_types = (int, long)
else:
    itervalues = lambda d: d.values()
    iteritems = lambda d: d.items()
    int_types = (int,)

__all__ = ["Neal", "NealSampler"]

class NealSampler(TemplateSampler):
    @ising(1, 2)
    def sample_ising(self, h, J, beta_range=None, num_samples=10, sweeps=1000,
                     beta_schedule_type="linear", seed=None):
        """
        Sample from low-energy spin states using simulated annealing 
        sampler written in C++.

        Args:
            h (dict): A dictionary of the linear biases in the Ising
                problem. Should be of the form {v: bias, ...} for each
                variable v in the Ising problem.
            J (dict): A dictionary of the quadratic biases in the Ising
                problem. Should be a dict of the form 
                {(u, v): bias, ...} for each edge (u, v) in the Ising 
                problem. If J[(u, v)] and J[(v, u)] exist then the 
                biases are added.
            beta_range (tuple, optional): A 2-tuple defining the
                beginning and end of the beta schedule (beta is the
                inverse temperature). The schedule is applied linearly
                in beta. Default is chosen based on the total bias 
                associated with each node.
            num_samples (int, optional): Each sample is the result of
                a single run of the simulated annealing algorithm.
            sweeps (int, optional): The number of sweeps or steps.
                Default is 1000.
            beta_schedule_type (string, optional): The beta schedule
                type, or how the beta values are interpolated between
                the given 'beta_range'. Default is "linear".
            seed (int, optional): The seed to use for the PRNG.
                Supplying the same seed with the rest of the same 
                parameters should produce identical results. Default
                is 'None', for which a random seed will be filled in.

        Returns:
            :obj:`SpinResponse`

        Examples:
            >>> sampler = FastSimulatedAnnealingSampler()
            >>> h = {0: -1, 1: -1}
            >>> J = {(0, 1): -1}
            >>> response = sampler.sample_ising(h, J, num_samples=1)
            >>> list(response.samples())
            [{0: 1, 1: 1}]

        Notes:
        ------
        This requires the GeneralSimulatedAnnealing Cython
        interface to the C++ CPU solver.

        A linear scheudule is used for beta.
        """

        # input checking
        # h, J are handled by the @ising decorator
        # beta_range, sweeps are handled by ising_simulated_annealing
        if not isinstance(num_samples, int):
            raise TypeError("'samples' should be a positive integer")
        if num_samples < 1:
            raise ValueError("'samples' should be a positive integer")
        if not (seed is None or isinstance(seed, int_types)):
            raise TypeError("'seed' should be None or a positive integer")
        if isinstance(seed, int_types) and not (0 < seed < (2**64 - 1)):
            error_msg = "'seed' should be an integer between 0 and 2^64 - 1"
            raise ValueError(error_msg)

        # @ising decorator ensures that all variables are in `h`
        var_map, vector_h = {}, []
        for new_var_index, (var, weight) in enumerate(iteritems(h)):
            var_map[var] = new_var_index
            vector_h.append(weight)
        
        if len(J) > 0:
            couplers, coupler_weights = zip(*iteritems(J))
            couplers = map(lambda c: (var_map[c[0]], var_map[c[1]]), couplers)
            coupler_starts, coupler_ends = zip(*couplers)
        else:
            coupler_starts, coupler_ends, coupler_weights = [], [], []

        if beta_range is None:
            beta_range = [.1]

            sigmas = {v: abs(h[v]) for v in h}
            for u, v in J:
                sigmas[u] += abs(J[(u, v)])
                sigmas[v] += abs(J[(u, v)])

            if len(sigmas) > 0:
                beta_range.append(2 * max(itervalues(sigmas)))
            else:
                # completely empty problem, so beta_range doesn't matter
                beta_range.append(1)

        sweeps_per_beta = max(1, sweeps//1000.0)
        num_betas = int(math.ceil(sweeps/sweeps_per_beta))
        if beta_schedule_type == "linear":
            # interpolate a linear beta schedule
            beta_step = (beta_range[1] - beta_range[0]) / float(num_betas)
            beta_schedule = [beta_range[0]+s*beta_step 
                                for s in range(num_betas)]
        elif beta_schedule_type == "geometric":
            beta_step = (beta_range[1]/beta_range[0])**(1.0/num_betas)
            beta_schedule = [beta_range[0]*beta_step**i 
                                for i in range(num_betas)]
        else:
            raise NotImplementedError("Beta schedule type {} not implemented"
                    .format(beta_schedule_type))

        if seed is None:
            # pick a random seed
            seed = randint(0, (1<<64 - 1))

        # run the simulated annealing algorithm
        samples, energies = simulated_annealing(num_samples, vector_h,
                                                coupler_starts, coupler_ends,
                                                coupler_weights, 
                                                sweeps_per_beta, beta_schedule,
                                                seed)

        # samples is now a 2d array with samples as row, exactly as 
        # dimod.NumpySpinResponse expects

        # create the response object
        response = NumpySpinResponse()

        # add the samples and corresponding energies
        response.add_samples_from_array(samples, energies)

        return response

Neal = NealSampler
