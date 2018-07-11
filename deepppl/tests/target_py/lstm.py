import torch
from torch import tensor
import pyro
import pyro.distributions as dist


def guide_rnn(input, n_characters, target):
    guide_rnn = {}
    ewl = pyro.param('ewl', randn())
    ews = pyro.param('ews', exp(randn()))
    guide_rnn['encoder.weight'] = dist.Normal(ewl, ews)
    gw1l = pyro.param('gw1l', randn())
    gw1s = pyro.param('gw1s', exp(randn()))
    guide_rnn['gru.weight_ih_l0'] = dist.Normal(gw1l, gw1s)
    gw2l = pyro.param('gw2l', randn())
    gw2s = pyro.param('gw2s', exp(randn()))
    guide_rnn['gru.weight_hh_l0'] = dist.Normal(gw2l, gw2s)
    gb1l = pyro.param('gb1l', randn())
    gb1s = pyro.param('gb1s', exp(randn()))
    guide_rnn['gru.bias_ih_l0'] = dist.Normal(gb1l, gb1s)
    gb2l = pyro.param('gb2l', randn())
    gb2s = pyro.param('gb2s', exp(randn()))
    guide_rnn['gru.bias_hh_l0'] = dist.Normal(gb2l, gb2s)
    dwl = pyro.param('dwl', randn())
    dws = pyro.param('dws', exp(randn()))
    guide_rnn['decoder.weight'] = dist.Normal(dwl, dws)
    dbl = pyro.param('dbl', randn())
    dbs = pyro.param('dbs', exp(randn()))
    guide_rnn['decoder.bias'] = dist.Normal(dbl, dbs)
    lifted_rnn = pyro.random_module('rnn', rnn, guide_rnn)
    return lifted_rnn()


def prior_rnn():
    prior_rnn = {}
    prior_rnn['encoder.weight'] = dist.Normal(tensor(0), tensor(1))
    prior_rnn['gru.weight_ih_l0'] = dist.Normal(tensor(0), tensor(1))
    prior_rnn['gru.weight_hh_l0'] = dist.Normal(tensor(0), tensor(1))
    prior_rnn['gru.bias_ih_l0'] = dist.Normal(tensor(0), tensor(1))
    prior_rnn['gru.bias_hh_l0'] = dist.Normal(tensor(0), tensor(1))
    prior_rnn['decoder.weight'] = dist.Normal(tensor(0), tensor(1))
    prior_rnn['decoder.bias'] = dist.Normal(tensor(0), tensor(1))
    lifted_rnn = pyro.random_module('rnn', rnn, prior_rnn)
    return lifted_rnn()


def model(input, n_characters, target):
    rnn = prior_rnn()
    logits = rnn(input)
    pyro.sample('target', dist.Categorical(logits), obs=target)