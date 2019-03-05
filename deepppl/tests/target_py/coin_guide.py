import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def guide_(x: 'int[10]'=None):
    alpha_q: 'real' = pyro.param('alpha_q', (0.0 + 10 - 0.0) * rand(()) + 0.0)
    beta_q: 'real' = pyro.param('beta_q', (0.0 + 10 - 0.0) * rand(()) + 0.0)
    theta: 'real' = pyro.sample('theta', dist.Beta(alpha_q, beta_q))


def model(x: 'int[10]'=None):
    theta: 'real' = pyro.sample('theta', dist.Uniform(0.0, 1.0))
    pyro.sample('theta' + '__1', dist.Beta(10.0, 10.0), obs=theta)
    for i in range(1, 10 + 1):
        pyro.sample('x' + '__{}'.format(i - 1) + '__2', dist.Bernoulli(theta),
            obs=x[i - 1])
