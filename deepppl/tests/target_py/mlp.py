import torch
from torch import tensor
import pyro
import pyro.distributions as dist


def guide_mlp(batch_size, imgs, labels):
    ___shape = {}
    ___shape['imgs'] = [tensor(28), tensor(28), batch_size]
    ___shape['labels'] = batch_size
    ___shape['l1wloc'] = mlp.l1.weight.shape
    ___shape['l1wscale'] = mlp.l1.weight.shape
    ___shape['l1bloc'] = mlp.l1.bias.shape
    ___shape['l1bscale'] = mlp.l1.bias.shape
    ___shape['l2wloc'] = mlp.l2.weight.shape
    ___shape['l2wscale'] = mlp.l2.weight.shape
    ___shape['l2bloc'] = mlp.l2.bias.shape
    ___shape['l2bscale'] = mlp.l2.bias.shape
    guide_mlp = {}
    l1wloc = pyro.param('l1wloc', randn(___shape['l1wloc']))
    l1wscale = pyro.param('l1wscale', randn(___shape['l1wscale']))
    guide_mlp['l1.weight'] = dist.Normal(l1wloc, softplus(l1wscale))
    l1bloc = pyro.param('l1bloc', randn(___shape['l1bloc']))
    l1bscale = pyro.param('l1bscale', randn(___shape['l1bscale']))
    guide_mlp['l1.bias'] = dist.Normal(l1bloc, softplus(l1bscale))
    l2wloc = pyro.param('l2wloc', randn(___shape['l2wloc']))
    l2wscale = pyro.param('l2wscale', randn(___shape['l2wscale']))
    guide_mlp['l2.weight'] = dist.Normal(l2wloc, softplus(l2wscale))
    l2bloc = pyro.param('l2bloc', randn(___shape['l2bloc']))
    l2bscale = pyro.param('l2bscale', randn(___shape['l2bscale']))
    guide_mlp['l2.bias'] = dist.Normal(l2bloc, softplus(l2bscale))
    lifted_mlp = pyro.random_module('mlp', mlp, guide_mlp)
    return lifted_mlp()


def prior_mlp(batch_size, imgs, labels):
    ___shape = {}
    ___shape['imgs'] = [tensor(28), tensor(28), batch_size]
    ___shape['labels'] = batch_size
    prior_mlp = {}
    prior_mlp['l1.weight'] = dist.Normal(zeros(mlp.l1.weight.shape), ones(
        mlp.l1.weight.shape))
    prior_mlp['l1.bias'] = dist.Normal(zeros(mlp.l1.bias.shape), ones(mlp.
        l1.bias.shape))
    prior_mlp['l2.weight'] = dist.Normal(zeros(mlp.l2.weight.shape), ones(
        mlp.l2.weight.shape))
    prior_mlp['l2.bias'] = dist.Normal(zeros(mlp.l2.bias.shape), ones(mlp.
        l2.bias.shape))
    lifted_mlp = pyro.random_module('mlp', mlp, prior_mlp)
    return lifted_mlp()


def model(batch_size, imgs, labels):
    ___shape = {}
    ___shape['imgs'] = [tensor(28), tensor(28), batch_size]
    ___shape['labels'] = batch_size
    mlp = prior_mlp(batch_size, imgs, labels)
    ___shape['logits'] = batch_size
    logits = mlp(imgs)
    pyro.sample('labels' + '1', CategoricalLogits(logits), obs=labels)
