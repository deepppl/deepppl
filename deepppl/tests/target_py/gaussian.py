import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def model():
    ___shape = {}
    ___shape['theta'] = ()
    theta = pyro.sample('theta', ImproperUniform())
    pyro.sample('theta' + '__1', dist.Normal(1000.0, 1.0), obs=theta)
    return {'theta': theta}
