import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def guide_(x: 'int[28,28]'=None):
    pyro.module('encoder', encoder)
    encoded: 'real[2,encoder(x)$shape[1]]' = encoder(x)
    mu_z: 'real[encoder(x)$shape[1]]' = encoded[1 - 1]
    sigma_z: 'real[encoder(x)$shape[1]]' = encoded[2 - 1]
    z: 'real[encoder(x)$shape[1]][??]' = pyro.sample('z', dist.Normal(mu_z,
        sigma_z))


def model(x: 'int[28,28]'=None):
    pyro.module('decoder', decoder)
    z: 'real[encoder(x)$shape[1]][??]' = pyro.sample('z', ImproperUniform())
    mu: 'real[28,28]' = zeros((28, 28))
    pyro.sample('z' + '__1', dist.Normal(0, 1), obs=z)
    mu: 'real[28,28]' = decoder(z)
    pyro.sample('x' + '__2', dist.Bernoulli(mu), obs=x)
