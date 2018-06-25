import torch
from torch import tensor
import pyro
import pyro.distributions as dist


def model(x):
    theta = pyro.sample('theta', dist.Uniform(tensor(0), tensor(1)))
    pyro.sample('x', dist.Bernoulli(theta), obs=x)