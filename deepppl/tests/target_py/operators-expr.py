import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def model(x):
    ___shape = {}
    ___shape['x'] = 10
    ___shape['theta'] = ()
    theta = pyro.sample('theta', dist.Uniform(0.0, 1.0))
    pyro.sample('theta' + '1', dist.Uniform(0 * 3 / 5, 1 + 5 - 5), obs=theta)
    for i in range(1, 10 + 1):
        if 1 <= 10 and (1 > 5 or 2 < 1):
            pyro.sample('x' + '{}'.format(i - 1) + '2', dist.Bernoulli(
                theta), obs=x[i - 1])
    print(x)
