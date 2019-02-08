
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import mcmc
import deepppl
import os
import numpy as np


def nuts(model, **kwargs):
    nuts_kernel = mcmc.NUTS(model, adapt_step_size=True)
    return mcmc.MCMC(nuts_kernel, **kwargs)


def test_linear_regression():
    model = deepppl.DppplModel(
        model_file='deepppl/tests/good/linear_regression.stan')
    posterior = model.posterior(
        method=nuts,
        num_samples=3000,
        warmup_steps=300)

    # Add Data
    num_samples = 100
    X = np.arange(num_samples)
    y = np.arange(num_samples)
    X = torch.Tensor(X)
    y = torch.Tensor(y)
    marginal = pyro.infer.EmpiricalMarginal(posterior.run(
        num_samples, X, y), sites=['alpha', 'beta', 'sigma'])
    samples_fstan = [marginal() for _ in range(30)]
    stack_samples = torch.stack(samples_fstan)
    params = torch.mean(stack_samples, 0) 
    print(params)


test_linear_regression()
