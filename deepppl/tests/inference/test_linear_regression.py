
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import mcmc
import deepppl
import os
import numpy as np
# from utils.utils import ImproperUniform

torch.manual_seed(11)

# def model(N, x, y):
#     alpha = pyro.sample('alpha', ImproperUniform())
#     beta = pyro.sample('beta', ImproperUniform())
#     sigma = pyro.sample('sigma', ImproperUniform())
#     pyro.sample('y' + '1', dist.Normal(alpha + beta * x, sigma), obs=y)


def model(N, x, y):
    alpha = pyro.sample('alpha', dist.Normal(0, 1000))
    beta = pyro.sample('beta', dist.Normal(0, 1000))
    sigma = pyro.sample('sigma', dist.Uniform(0, 1000))
    pyro.sample('y' + '1', dist.Normal(alpha + beta * x, sigma), obs=y)


def nuts(model, **kwargs):
    nuts_kernel = mcmc.NUTS(model, adapt_step_size=True)
    return mcmc.MCMC(nuts_kernel, **kwargs)


def test_linear_regression():
    # model = deepppl.DppplModel(model_file = 'deepppl/tests/good/linear_regression.stan')
    #posterior = model.posterior(method=nuts, num_samples=3000, warmup_steps=300)
    posterior = nuts(model, num_samples=1000, warmup_steps=200)

    #hmc_posterior = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200) \
    # .run(is_cont_africa, ruggedness, log_gdp)
    # Add Data
    num_samples = 100
    X = np.arange(num_samples)
    y = np.arange(num_samples)
    X = torch.Tensor(X)
    y = torch.Tensor(y)
    marginal = pyro.infer.EmpiricalMarginal(posterior.run(
        num_samples, X, y), sites=['alpha', 'beta', 'sigma'])
    # samples_fstan = [marginal().item() for _ in range(3000)]
    # print(samples_fstan)


test_linear_regression()
