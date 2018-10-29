import torch
from torch import tensor
import pyro
import pyro.distributions as dist


def model():
    ___shape = {}
    theta = pyro.sample('theta', ImproperUniform())
    pyro.sample('theta' + '1', dist.Normal(1000.0, 1.0), obs=theta)
    pyro.sample('theta' + '2', dist.Normal(1000.0, 1.0), obs=theta)
