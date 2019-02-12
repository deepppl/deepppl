import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def guide_(nz=None, x=None):
    ___shape = {}
    ___shape['nz'] = ()
    ___shape['x'] = [28, 28]
    ___shape['z'] = nz
    pyro.module('encoder', encoder)
    ___shape['encoded'] = [2, nz]
    encoded = encoder(x)
    ___shape['mu_z'] = nz
    mu_z = encoded[1 - 1]
    ___shape['sigma_z'] = nz
    sigma_z = encoded[2 - 1]
    z = pyro.sample('z', dist.Normal(mu_z, sigma_z))


def model(nz=None, x=None):
    ___shape = {}
    ___shape['nz'] = ()
    ___shape['x'] = [28, 28]
    ___shape['z'] = nz
    pyro.module('decoder', decoder)
    z = pyro.sample('z', ImproperUniform(nz))
    ___shape['mu'] = [28, 28]
    mu = zeros(___shape['mu'])
    pyro.sample('z' + '1', dist.Normal(zeros(nz), ones(nz)), obs=z)
    mu = decoder(z)
    pyro.sample('x' + '2', dist.Bernoulli(mu), obs=x)
