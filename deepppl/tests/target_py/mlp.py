import torch
import pyro
import pyro.distributions as dist


def guide_mlp(batch_size, imgs, labels):
    guide_mlp = {}
    l1wloc = pyro.param('l1wloc', dist.Normal(0, 1).sample())
    l1wscale = pyro.param('l1wscale', dist.Normal(0, 1).sample())
    guide_mlp['l1.weight'] = dist.Normal(l1wloc, l1wscale)
    l1bloc = pyro.param('l1bloc', dist.Normal(0, 1).sample())
    l1bscale = pyro.param('l1bscale', dist.Normal(0, 1).sample())
    guide_mlp['l1.bias'] = dist.Normal(l1bloc, l1bscale)
    l2wloc = pyro.param('l2wloc', dist.Normal(0, 1).sample())
    l2wscale = pyro.param('l2wscale', dist.Normal(0, 1).sample())
    guide_mlp['l2.weight'] = dist.Normal(l2wloc, l2wscale)
    l2bloc = pyro.param('l2bloc', dist.Normal(0, 1).sample())
    l2bscale = pyro.param('l2bscale', dist.Normal(0, 1).sample())
    guide_mlp['l2.bias'] = dist.Normal(l2bloc, l2bscale)
    lifted_mlp = pyro.random_module('mlp', mlp, guide_mlp)
    return lifted_mlp()


def prior_mlp():
    prior_mlp = {}
    prior_mlp['l1.weight'] = dist.Normal(0, 1)
    prior_mlp['l1.bias'] = dist.Normal(0, 1)
    prior_mlp['l2.weight'] = dist.Normal(0, 1)
    prior_mlp['l2.bias'] = dist.Normal(0, 1)
    lifted_mlp = pyro.random_module('mlp', mlp, prior_mlp)
    return lifted_mlp()


def model(batch_size, imgs, labels):
    mlp = prior_mlp()
    logits = mlp(imgs)
    pyro.sample('labels', dist.Categorical(logits), obs=labels)