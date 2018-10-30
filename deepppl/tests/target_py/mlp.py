import torch
from torch import tensor, randn
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def guide_mlp(batch_size, imgs, labels):
    ___shape = {}
    ___shape['batch_size'] = ()
    ___shape['imgs'] = [28, 28, batch_size]
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
    ___shape['batch_size'] = ()
    ___shape['imgs'] = [28, 28, batch_size]
    ___shape['labels'] = batch_size
    prior_mlp = {}
    prior_mlp['l1.weight'] = ImproperUniform(mlp.l1.weight.shape)
    prior_mlp['l1.bias'] = ImproperUniform(mlp.l1.bias.shape)
    prior_mlp['l2.weight'] = ImproperUniform(mlp.l2.weight.shape)
    prior_mlp['l2.bias'] = ImproperUniform(mlp.l2.bias.shape)
    lifted_mlp = pyro.random_module('mlp', mlp, prior_mlp)
    return lifted_mlp()


def model(batch_size, imgs, labels):
    ___shape = {}
    ___shape['batch_size'] = ()
    ___shape['imgs'] = [28, 28, batch_size]
    ___shape['labels'] = batch_size
    mlp = prior_mlp(batch_size, imgs, labels)
    model_mlp = mlp.state_dict()
    ___shape['logits'] = batch_size
    pyro.sample('model_mlp' + '{}'.format('l1.weight') + '1', dist.Normal(
        zeros(mlp.l1.weight.shape), ones(mlp.l1.weight.shape)), obs=
        model_mlp['l1.weight'])
    pyro.sample('model_mlp' + '{}'.format('l1.bias') + '2', dist.Normal(
        zeros(mlp.l1.bias.shape), ones(mlp.l1.bias.shape)), obs=model_mlp[
        'l1.bias'])
    pyro.sample('model_mlp' + '{}'.format('l2.weight') + '3', dist.Normal(
        zeros(mlp.l2.weight.shape), ones(mlp.l2.weight.shape)), obs=
        model_mlp['l2.weight'])
    pyro.sample('model_mlp' + '{}'.format('l2.bias') + '4', dist.Normal(
        zeros(mlp.l2.bias.shape), ones(mlp.l2.bias.shape)), obs=model_mlp[
        'l2.bias'])
    logits = mlp(imgs)
    pyro.sample('labels' + '5', CategoricalLogits(logits), obs=labels)