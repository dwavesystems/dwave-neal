.. intro_neal:

============
Introduction
============

*Samplers* are processes that sample from low energy states of a problem’s objective function.
A binary quadratic model (:term:`BQM`\ ) :term:`sampler` samples from low energy states in models
such as those defined by an :term:`Ising` equation or a Quadratic Unconstrained Binary Optimization
(:term:`QUBO`) problem and returns an iterable of samples, in order of increasing energy. A
:std:doc:`dimod <oceandocs:docs_dimod/sdk_index>` sampler provides ‘sample_qubo’ and
‘sample_ising’ methods as well as the generic BQM sampler method.

The :class:`~neal.SimulatedAnnealingSampler` sampler implements the simulated annealing
algorithm, based on the technique of cooling metal from a high temperature to improve its
structure (annealing). This algorithm often finds good solutions to hard optimization problems.
