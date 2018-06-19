import torch
import pyro
import pyro.distributions as dist


def guide_(x):
    alpha_q = pyro.param('alpha_q', 15)
    beta_q = pyro.param('beta_q', 15)
    pyro.sample("theta", dist.Beta(alpha_q, beta_q))


def model(x):
    theta = pyro.sample('theta', dist.Beta(10, 10))
    for i in range(1, 10 + 1):
        pyro.sample('x' + '{}'.format(i - 1), dist.Bernoulli(theta), obs=x[
            i - 1])