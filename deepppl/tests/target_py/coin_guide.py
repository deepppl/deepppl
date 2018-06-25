import torch
from torch import tensor
import pyro
import pyro.distributions as dist


def guide_(x):
    alpha_q = pyro.param('alpha_q', tensor(15.0))
    beta_q = pyro.param('beta_q', tensor(15.0))
    theta = pyro.sample('theta', dist.Beta(alpha_q, beta_q))


def model(x):
    theta = pyro.sample('theta', dist.Beta(tensor(10.0), tensor(10.0)))
    for i in range(tensor(1), tensor(10) + 1):
        pyro.sample('x' + '{}'.format(i - 1), dist.Bernoulli(theta), obs=x[
            i - 1])