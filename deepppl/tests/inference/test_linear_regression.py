
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import mcmc
import deepppl
import os
import numpy as np

from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.contrib.autoguide import AutoDiagonalNormal


def nuts(model, **kwargs):
    nuts_kernel = mcmc.NUTS(model, adapt_step_size=True)
    return mcmc.MCMC(nuts_kernel, **kwargs)


import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist
from deepppl.utils.utils import ImproperUniform, LowerConstrainedImproperUniform
import sys


# def model2(N, x, y):
#     alpha = pyro.sample('alpha', dist.Normal(0, 1))
#     beta = pyro.sample('beta', dist.Normal(0, 1))
#     sigma = pyro.sample('sigma', dist.Uniform(0, 10))
#     # with pyro.iarange('data', N):
#     pyro.sample('y' + '1', dist.Normal(alpha + beta * x, sigma), obs=y)


# def guide(N, x, y):
#     a_loc = pyro.param('a_loc', torch.tensor(0.))
#     a_scale = pyro.param('a_scale', torch.tensor(
#         1.), constraint=constraints.positive)
#     b_loc = pyro.param('b_loc', torch.tensor(0.))
#     b_scale = pyro.param('b_scale', torch.tensor(
#         1.), constraint=constraints.positive)
#     s_loc = pyro.param('s_loc', torch.tensor(
#         1.), constraint=constraints.positive)
#     alpha = pyro.sample('alpha', dist.Normal(a_loc, a_scale))
#     beta = pyro.sample('beta', dist.Normal(b_loc, b_scale))
#     sigma = pyro.sample('sigma', dist.Normal(s_loc, torch.tensor(0.05)))


def test_linear_regression():
    model = deepppl.DppplModel(
        model_file='deepppl/tests/good/linear_regression.stan')

    # model._model = model2

    # model = dmodel._model
    # guide = AutoDiagonalNormal(model)

    # Add Data
    num_samples = 50
    X = np.arange(num_samples)
    y = np.arange(num_samples)
    X = torch.Tensor(X)
    y = torch.Tensor(y)

    # optim = Adam({"lr": 0.03})
    # svi = SVI(model, guide, optim, loss=Trace_ELBO(), num_samples=1000)

    # for step in range(5000):
    #     svi.step(num_samples, X, y)
    #     if step % 100 == 0:
    #         print('.', end='')
    #         sys.stdout.flush()

    posterior = model.posterior(
        method=nuts,
        num_samples=3000,
        warmup_steps=300)

    marginal = pyro.infer.EmpiricalMarginal(
        posterior.run(num_samples, X, y),
        sites=['alpha', 'beta', 'sigma'])
    samples_fstan = [marginal() for _ in range(30)]
    stack_samples = torch.stack(samples_fstan)
    params = torch.mean(stack_samples, 0)
    # alpha = pyro.param("a_loc").item()
    # beta = pyro.param("b_loc").item()
    # sigma = pyro.param("s_loc").item()

    # print(alpha, beta, sigma)

    # for name, value in pyro.get_param_store():
    #     print(name, pyro.param(name), value)

    # print(params)


test_linear_regression()
