import torch
from torch import tensor
import pyro
import pyro.distributions as dist


def guide_(batch_size, nz, x):
    ___shape = {}
    pyro.module('encoder', encoder)
    ___shape['encoded'] = tensor(2)
    encoded = encoder(x)
    mu = encoded[tensor(1) - 1]
    sigma = encoded[tensor(2) - 1]
    latent = pyro.sample('latent', dist.Normal(mu, sigma, batch_size))


def model(batch_size, nz, x):
    ___shape = {}
    pyro.module('decoder', decoder)
    latent = pyro.sample('latent', ImproperUniform())
    pyro.sample('latent' + '1', dist.Normal(zeros(nz), ones(nz), batch_size), obs=latent)
    loc_img = decoder(latent)
    pyro.sample('x' + '2', dist.Bernoulli(loc_img), obs=x)