import torch
from torch import tensor
import pyro
import pyro.distributions as dist


def model(x):
    theta = pyro.sample('theta', dist.Uniform(tensor(0.0), tensor(1.0)))
    for i in range(tensor(1), tensor(10) + 1):
        pyro.sample('x' + '{}'.format(i - 1), dist.Bernoulli(theta), obs=x[i - 1])