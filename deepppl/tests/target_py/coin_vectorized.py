import torch
import pyro
import pyro.distributions as dist


def model(x):
    theta = pyro.sample('theta', dist.Uniform(0, 1))
    pyro.sample('x', dist.Bernoulli(theta), obs=x)