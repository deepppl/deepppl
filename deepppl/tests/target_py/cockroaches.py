import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def transformed_data(N=None, exposure2=None, roach1=None, senior=None,
    treatment=None, y=None):
    log_expo = zeros(N)
    sqrt_roach = zeros(N)
    log_expo = log(exposure2)
    sqrt_roach = sqrt(roach1)
    return {'log_expo': log_expo, 'sqrt_roach': sqrt_roach}


def model(N=None, exposure2=None, roach1=None, senior=None, treatment=None,
    y=None, transformed_data=None):
    sqrt_roach = transformed_data['sqrt_roach']
    log_expo = transformed_data['log_expo']
    beta = sample('beta', ImproperUniform(shape=4))
    sample('beta' + '__{}'.format(1 - 1) + '__1', dist.Normal(0, 5), obs=
        beta[1 - 1])
    sample('beta' + '__{}'.format(2 - 1) + '__2', dist.Normal(0, 2.5), obs=
        beta[2 - 1])
    sample('beta' + '__{}'.format(3 - 1) + '__3', dist.Normal(0, 2.5), obs=
        beta[3 - 1])
    sample('beta' + '__{}'.format(4 - 1) + '__4', dist.Normal(0, 2.5), obs=
        beta[4 - 1])
    sample('y' + '__5', poisson_log(log_expo + beta[1 - 1] + beta[2 - 1] *
        sqrt_roach + beta[3 - 1] * treatment + beta[4 - 1] * senior), obs=y)
