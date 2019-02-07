import torch
from torch import tensor, randn
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def model(M, N, x, y):
    ___shape = {}
    ___shape['N'] = ()
    ___shape['M'] = ()
    ___shape['y'] = N
    ___shape['x'] = N
    ___shape['beta'] = ()
    beta = pyro.sample('beta', ImproperUniform())
    for m in range(1, M + 1):
        pyro.sample('beta' + '{}'.format(m - 1) + '1', dist.Cauchy(0.0, 2.5
            ), obs=beta[m - 1])
    for n in range(1, N + 1):
        pyro.sample('y' + '{}'.format(n - 1) + '2', dist.Bernoulli(
            inv_logit(x[n - 1] * beta)), obs=y[n - 1])
