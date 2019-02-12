import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def model(K=None, M=None, N=None, V=None, alpha=None, beta=None, doc=None,
    w=None):
    ___shape = {}
    ___shape['K'] = ()
    ___shape['V'] = ()
    ___shape['M'] = ()
    ___shape['N'] = ()
    ___shape['w'] = N
    ___shape['doc'] = N
    ___shape['alpha'] = K
    ___shape['beta'] = V
    ___shape['theta'] = M, K
    ___shape['phi'] = K, V
    theta = pyro.sample('theta', ImproperUniform((M, K)))
    phi = pyro.sample('phi', ImproperUniform((K, V)))
    for m in range(1, M + 1):
        pyro.sample('theta' + '{}'.format(m - 1) + '1', dist.Dirichlet(
            alpha), obs=theta[m - 1])
    for k in range(1, K + 1):
        pyro.sample('phi' + '{}'.format(k - 1) + '2', dist.Dirichlet(beta),
            obs=phi[k - 1])
    for n in range(1, N + 1):
        ___shape['gamma'] = K
        gamma = zeros(___shape['gamma'])
        for k in range(1, K + 1):
            gamma[k - 1] = log(theta[[doc[n - 1], k] - 1]) + log(phi[[k, w[
                n - 1]] - 1])
        pyro.sample('expr' + '{}'.format(n) + '3', dist.Exponential(1.0),
            obs=-log_sum_exp(gamma))
