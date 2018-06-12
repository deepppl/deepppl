import torch
import pyro
import pyro.distributions as dist


def model(x):
    theta = pyro.sample('theta', dist.Uniform(0, 1))
    for i in range(1, 10 + 1):
        pyro.sample('x' + '{}'.format(i - 1), dist.Bernoulli(theta), obs=x[i - 1])