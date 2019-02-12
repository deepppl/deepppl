import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def transformed_data(D=None, K=None, N=None, y=None):
    ___shape = {}
    ___shape['neg_log_K'] = ()
    neg_log_K = -log(K)
    return {'neg_log_K': neg_log_K}


def model(D=None, K=None, N=None, y=None, transformed_data=None):
    neg_log_K = transformed_data['neg_log_K']
    ___shape = {}
    ___shape['N'] = ()
    ___shape['D'] = ()
    ___shape['K'] = ()
    ___shape['y'] = N
    ___shape['mu'] = K
    mu = pyro.sample('mu', ImproperUniform(K))
    ___shape['soft_z'] = [N, K]
    soft_z = zeros(___shape['soft_z'])
    for n in range(1, N + 1):
        for k in range(1, K + 1):
            soft_z[n - 1, k - 1] = neg_log_K - 0.5 * dot_self(mu[k - 1] - y
                [n - 1])
    for k in range(1, K + 1):
        pyro.sample('mu' + '{}'.format(k - 1) + '1', dist.Normal(0, 1), obs
            =mu[k - 1])
    for n in range(1, N + 1):
        pyro.sample('expr' + '2', dist.Exponential(1.0), obs=-log_sum_exp(
            soft_z[n - 1]))
