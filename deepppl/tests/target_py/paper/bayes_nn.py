import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist


def guide_mlp(img: 'int[28,28]'=None, label: 'int'=None):
    guide_mlp = {}
    w1_mu: 'real' = pyro.param('w1_mu', (2 - -2) * rand(()) + -2)
    w1_sgma: 'real' = pyro.param('w1_sgma', (2 - -2) * rand(()) + -2)
    b1_mu: 'real' = pyro.param('b1_mu', (2 - -2) * rand(()) + -2)
    b1_sgma: 'real' = pyro.param('b1_sgma', (2 - -2) * rand(()) + -2)
    w2_mu: 'real' = pyro.param('w2_mu', (2 - -2) * rand(()) + -2)
    w2_sgma: 'real' = pyro.param('w2_sgma', (2 - -2) * rand(()) + -2)
    b2_mu: 'real' = pyro.param('b2_mu', (2 - -2) * rand(()) + -2)
    b2_sgma: 'real' = pyro.param('b2_sgma', (2 - -2) * rand(()) + -2)
    guide_mlp['l1.weight']: 'real' = dist.Normal(w1_mu, exp(w1_sgma))
    guide_mlp['l1.bias']: 'real' = dist.Normal(b1_mu, exp(b1_sgma))
    guide_mlp['l2.weight']: 'real' = dist.Normal(w2_mu, exp(w2_sgma))
    guide_mlp['l2.bias']: 'real' = dist.Normal(b2_mu, exp(b2_sgma))
    lifted_mlp = pyro.random_module('mlp', mlp, guide_mlp)
    return lifted_mlp()


def prior_mlp(img: 'int[28,28]'=None, label: 'int'=None):
    prior_mlp = {}
    prior_mlp['l1.weight']: 'real' = ImproperUniform()
    prior_mlp['l1.bias']: 'real' = ImproperUniform()
    prior_mlp['l2.weight']: 'real' = ImproperUniform()
    prior_mlp['l2.bias']: 'real' = ImproperUniform()
    lifted_mlp = pyro.random_module('mlp', mlp, prior_mlp)
    return lifted_mlp()


def model(img: 'int[28,28]'=None, label: 'int'=None):
    mlp = prior_mlp(img, label)
    model_mlp = mlp.state_dict()
    logits: 'real[10]' = zeros(10)
    pyro.sample('model_mlp' + '__{}'.format('l1.weight') + '__1', dist.
        Normal(0, 1), obs=model_mlp['l1.weight'])
    pyro.sample('model_mlp' + '__{}'.format('l1.bias') + '__2', dist.Normal
        (0, 1), obs=model_mlp['l1.bias'])
    pyro.sample('model_mlp' + '__{}'.format('l2.weight') + '__3', dist.
        Normal(0, 1), obs=model_mlp['l2.weight'])
    pyro.sample('model_mlp' + '__{}'.format('l2.bias') + '__4', dist.Normal
        (0, 1), obs=model_mlp['l2.bias'])
    logits: 'real[10]' = mlp(img)
    pyro.sample('label' + '__5', categorical_logits(logits), obs=label)
