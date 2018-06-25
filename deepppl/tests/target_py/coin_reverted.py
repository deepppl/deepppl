import torch
from torch import tensor
import pyro
import pyro.distributions as dist


def model(x):
    theta = pyro.sample('theta', dist.Uniform(torch(0.0), torch(1.0)))
    pyro.sample('x', dist.Bernoulli(theta), obs=x)