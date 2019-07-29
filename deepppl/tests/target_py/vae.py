import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def guide_(nz: 'int'=None, x: 'int[28,28]'=None):
    pyro.module('encoder', encoder)
    encoded: 'real[2,nz]' = encoder(x)
    mu_z: 'real[nz]' = encoded[1 - 1]
    sigma_z: 'real[nz]' = encoded[2 - 1]
    z: 'real[nz][??]' = sample('z', dist.Normal(mu_z, sigma_z))


def model(nz: 'int'=None, x: 'int[28,28]'=None):
    pyro.module('decoder', decoder)
    z: 'real[nz][??]' = sample('z', ImproperUniform(shape=nz))
    mu: 'real[28,28]' = zeros((28, 28))
    sample('z' + '__1', dist.Normal(zeros(nz), 1), obs=z)
    mu: 'real[28,28]' = decoder(z)
    sample('x' + '__2', dist.Bernoulli(mu), obs=x)
