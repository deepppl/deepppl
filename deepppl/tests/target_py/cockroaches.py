import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def transformed_data(N: 'int'=None, exposure2: 'real[N]'=None, roach1:
    'real[N]'=None, senior: 'real[N]'=None, treatment: 'real[N]'=None, y:
    'int[N]'=None):
    log_expo: 'real[N]' = zeros(N)
    log_expo: 'real[N]' = log(exposure2)
    return {'log_expo': log_expo}


def model(N: 'int'=None, exposure2: 'real[N]'=None, roach1: 'real[N]'=None,
    senior: 'real[N]'=None, treatment: 'real[N]'=None, y: 'int[N]'=None,
    transformed_data=None):
    log_expo = transformed_data['log_expo']
    beta: 'real[4]' = sample('beta', ImproperUniform(shape=4))
    lmbda: 'real[N]' = sample('lmbda', ImproperUniform(shape=N))
    tau: 'real' = sample('tau', ImproperUniform())
    sigma: 'real' = 1.0 / sqrt(tau)
    sample('tau' + '__1', dist.Gamma(0.001, 0.001), obs=tau)
    sample('lmbda' + '__2', dist.Normal(zeros(N), sigma), obs=lmbda)
    sample('y' + '__3', poisson_log(log_expo + beta[1 - 1] + beta[2 -
        1] * roach1 + beta[3 - 1] * treatment + beta[4 - 1] * senior +
        lmbda), obs=y)


def generated_quantities(N: 'int'=None, exposure2: 'real[N]'=None, roach1:
    'real[N]'=None, senior: 'real[N]'=None, treatment: 'real[N]'=None, y:
    'int[N]'=None, transformed_data=None, parameters=None):
    log_expo = transformed_data['log_expo']
    beta = parameters['beta']
    lmbda = parameters['lmbda']
    tau = parameters['tau']
    sigma: 'real' = 1.0 / sqrt(tau)
    return {'sigma': sigma}
