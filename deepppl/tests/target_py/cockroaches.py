import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def transformed_data(N=None, exposure2=None, roach1=None, senior=None,
    treatment=None, y=None):
    log_expo = zeros(N)
    log_expo = log(exposure2)
    return {'log_expo': log_expo}


def model(N=None, exposure2=None, roach1=None, senior=None, treatment=None,
    y=None, transformed_data=None):
    log_expo = transformed_data['log_expo']
    beta = sample('beta', ImproperUniform(shape=4))
    lambda_ = sample('lambda_', ImproperUniform(shape=N))
    tau = sample('tau', LowerConstrainedImproperUniform(0.0))
    sigma = 1.0 / sqrt(tau)
    sample('tau' + '__1', dist.Gamma(0.001, 0.001), obs=tau)
    sample('lambda_' + '__2', dist.Normal(zeros(N), sigma), obs=lambda_)
    sample('y' + '__3', poisson_log(lambda_ + log_expo + beta[1 - 1] + beta
        [2 - 1] * roach1 + beta[3 - 1] * senior + beta[4 - 1] * treatment),
        obs=y)


def generated_quantities(N=None, exposure2=None, roach1=None, senior=None,
    treatment=None, y=None, transformed_data=None, parameters=None):
    log_expo = transformed_data['log_expo']
    beta = parameters['beta']
    lambda_ = parameters['lambda_']
    tau = parameters['tau']
    sigma = 1.0 / sqrt(tau)
    return {'sigma': sigma}