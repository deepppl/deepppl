import torch
from torch import tensor
import pyro
import pyro.distributions as dist


def guide_(x):
    ___shape = {}
    pyro.module('encoder', encoder)
    ___shape['encoded'] = tensor(2)
    encoded = encoder(x)
    mu = encoded[tensor(1) - 1]
    sigma = encoded[tensor(2) - 1]
    latent = pyro.sample('latent', dist.Normal(mu, sigma))


def model(x):
    ___shape = {}
    pyro.module("decoder", decoder)
    latent = pyro.sample('latent', dist.Normal(tensor(0), tensor(1)))
    loc_img = decoder(latent)
    pyro.sample('x', dist.Bernoulli(loc_img), obs=x)