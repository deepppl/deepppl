import torch
from torch import tensor, randn
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def transformed_data(N, x):
    ___shape = {}
    ___shape['K'] = ()
    K = zeros(___shape['K'])
    ___shape['mu'] = ()
    mu = rep_vector(0, N)
    for i in range(1, N - 1 + 1):
        K[i - 1, i - 1] = 1 + 0.1
        for j in range(i + 1, N + 1):
            K[i - 1, j - 1] = exp(-0.5 * square(x[i - 1] - x[j - 1]))
            K[j - 1, i - 1] = K[[i, j] - 1]
    K[N - 1, N - 1] = 1 + 0.1
    return {'mu': mu, 'K': K}


def model(N, x, transformed_data):
    mu = transformed_data['mu']
    K = transformed_data['K']
    ___shape = {}
    ___shape['N'] = ()
    ___shape['x'] = N
    ___shape['y'] = ()
    y = pyro.sample('y', ImproperUniform())
    pyro.sample('y' + '1', dist.MultivariateNormal(mu, K), obs=y)