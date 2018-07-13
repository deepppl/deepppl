import torch
from torch import tensor
import pyro
import pyro.distributions as dist


def guide_rnn(input, n_characters, target):
    ___shape = {}
    ___shape['input'] = n_characters
    ___shape['target'] = n_characters
    ___shape['ewl'] = rnn.encoder.weight.shape()
    ___shape['ews'] = rnn.encoder.weight.shape()
    ___shape['gw1l'] = rnn.gru.weight_ih_l0.shape()
    ___shape['gw1s'] = rnn.gru.weight_ih_l0.shape()
    ___shape['gw2l'] = rnn.gru.weight_hh_l0.shape()
    ___shape['gw2s'] = rnn.gru.weight_hh_l0.shape()
    ___shape['gb1l'] = rnn.gru.bias_ih_l0.shape()
    ___shape['gb1s'] = rnn.gru.bias_ih_l0.shape()
    ___shape['gb2l'] = rnn.gru.bias_hh_l0.shape()
    ___shape['gb2s'] = rnn.gru.bias_hh_l0.shape()
    ___shape['dwl'] = rnn.decoder.weight.shape()
    ___shape['dws'] = rnn.decoder.weight.shape()
    ___shape['dbl'] = rnn.decoder.bias.shape()
    ___shape['dbs'] = rnn.decoder.bias.shape()
    guide_rnn = {}
    ewl = pyro.param('ewl', randn(___shape['ewl']))
    ews = pyro.param('ews', exp(randn(___shape['ews'])))
    guide_rnn['encoder.weight'] = dist.LogNormal(ewl, ews)
    gw1l = pyro.param('gw1l', randn(___shape['gw1l']))
    gw1s = pyro.param('gw1s', exp(randn(___shape['gw1s'])))
    guide_rnn['gru.weight_ih_l0'] = dist.Normal(gw1l, gw1s)
    gw2l = pyro.param('gw2l', randn(___shape['gw2l']))
    gw2s = pyro.param('gw2s', exp(randn(___shape['gw2s'])))
    guide_rnn['gru.weight_hh_l0'] = dist.Normal(gw2l, gw2s)
    gb1l = pyro.param('gb1l', randn(___shape['gb1l']))
    gb1s = pyro.param('gb1s', exp(randn(___shape['gb1s'])))
    guide_rnn['gru.bias_ih_l0'] = dist.Normal(gb1l, gb1s)
    gb2l = pyro.param('gb2l', randn(___shape['gb2l']))
    gb2s = pyro.param('gb2s', exp(randn(___shape['gb2s'])))
    guide_rnn['gru.bias_hh_l0'] = dist.Normal(gb2l, gb2s)
    dwl = pyro.param('dwl', randn(___shape['dwl']))
    dws = pyro.param('dws', exp(randn(___shape['dws'])))
    guide_rnn['decoder.weight'] = dist.Normal(dwl, dws)
    dbl = pyro.param('dbl', randn(___shape['dbl']))
    dbs = pyro.param('dbs', exp(randn(___shape['dbs'])))
    guide_rnn['decoder.bias'] = dist.Normal(dbl, dbs)
    lifted_rnn = pyro.random_module('rnn', rnn, guide_rnn)
    return lifted_rnn()


def prior_rnn(input, n_characters, target):
    ___shape = {}
    ___shape['input'] = n_characters
    ___shape['target'] = n_characters
    prior_rnn = {}
    prior_rnn['encoder.weight'] = dist.Normal(zeros(rnn.encoder.weight.shape()), ones(rnn.encoder.weight.shape()))
    prior_rnn['gru.weight_ih_l0'] = dist.Normal(zeros(rnn.gru.weight_ih_l0.shape()), ones(rnn.gru.weight_ih_l0.shape()))
    prior_rnn['gru.weight_hh_l0'] = dist.Normal(zeros(rnn.gru.weight_hh_l0.shape()), ones(rnn.gru.weight_hh_l0.shape()))
    prior_rnn['gru.bias_ih_l0'] = dist.Normal(zeros(rnn.gru.bias_ih_l0.shape()), ones(rnn.gru.bias_ih_l0.shape()))
    prior_rnn['gru.bias_hh_l0'] = dist.Normal(zeros(rnn.gru.bias_hh_l0.shape()), ones(rnn.gru.bias_hh_l0.shape()))
    prior_rnn['decoder.weight'] = dist.Normal(zeros(rnn.decoder.weight.shape()), ones(rnn.decoder.weight.shape()))
    prior_rnn['decoder.bias'] = dist.Normal(zeros(rnn.decoder.bias.shape()), ones(rnn.decoder.bias.shape()))
    lifted_rnn = pyro.random_module('rnn', rnn, prior_rnn)
    return lifted_rnn()


def model(input, n_characters, target):
    ___shape = {}
    ___shape['input'] = n_characters
    ___shape['target'] = n_characters
    rnn = prior_rnn(input, n_characters, target)
    ___shape['logits'] = n_characters
    logits = rnn(input)
    pyro.sample('target', dist.Categorical(logits), obs=target)