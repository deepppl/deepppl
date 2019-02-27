import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def model(x=None):
    ___shape = {}
    ___shape['x'] = 10
    ___shape['theta'] = ()
    theta = pyro.sample('theta', dist.Uniform(0.0, 1.0))
    pyro.sample('theta' + '__1', dist.Uniform(0, 1), obs=theta)
    for i in range(1, 10 + 1):
        pyro.sample('x' + '__{}'.format(i - 1) + '__2', dist.Bernoulli(theta),
            obs=x[i - 1])
