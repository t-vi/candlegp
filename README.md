# CandleGP - Gaussian Processes in Pytorch
*Thomas Viehmann*, tv@lernapparat.de

I felt Bayesian the other day, so I ported (a tiny part of) the
excellent [GPFlow library](https://github.com/gpflow/gpflow) to
Pytorch.

I have kept the structure close to GPFlow.
Most of the great ideas are from the GPFlow team, errors are my own.

The functionality is demonstrated by the following notebooks
adapted from gpflow:

- [Classification example](notebooks/classification.ipynb)
- [Regression example](notebooks/gp_regression.ipynb)
- [Markov Chain Monte Carlo for non-gaussian likelihoods](notebooks/mcmc.ipynb)
- [Upper bound for variational inference](notebooks/upper_bound.ipynb)
- [SVGP with minibatch training](notebooks/minibatches.ipynb)
