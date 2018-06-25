import torch
from torch import tensor
import pyro
import pyro.distributions as dist


def model(x):
    theta = pyro.sample('theta', dist.Uniform(tensor(0) * tensor(3) / tensor(5), tensor(1) + tensor(5) - tensor(5)))
    for i in range(tensor(1), tensor(10) + 1):
        if tensor(1) <= tensor(10) and (tensor(1) > tensor(5) or tensor(2) < tensor(1)):
            pyro.sample('x' + '{}'.format(i - 1), dist.Bernoulli(theta), obs=x[i - 1])
    print(x)