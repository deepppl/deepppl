import torch
import pyro
import pyro.distributions as dist

mlp = MLP()

def model(data):
    priors = {
        'l1.weight': Normal(torch.zeros(nh, nx), torch.ones(nh, nx)),
        'l1.bias': Normal(torch.zeros(nh), torch.ones(nh)),
        'l2.weight': Normal(torch.zeros(ny, nh), torch.ones(ny, nh)),
        'l2.bias': Normal(torch.zeros(ny), torch.ones(ny))}
    lifted_module = pyro.random_module("mlp", mlp, priors)
    lifted_reg_model = lifted_module()
    x, y = data
    yhat = F.log_softmax(lifted_reg_model(x))
    pyro.sample("obs", Categorical(logits=yhat), obs=y)

# Inference Guide

def vr(name, *shape):
    return pyro.param(name,
                        Variable(torch.randn(*shape), requires_grad=True))

def guide(data):
    dists = {
        'l1.weight': Normal(vr("W1m", nh, nx), F.softplus(vr("W1s", nh, nx))),
        'l1.bias': Normal(vr("b1m", nh), F.softplus(vr("b1s", nh))),
        'l2.weight': Normal(vr("W2m", ny, nh), F.softplus(vr("W2s", ny, nh))),
        'l2.bias': Normal(vr("b2m", ny), F.softplus(vr("b2s", ny)))}
    lifted_module = pyro.random_module("mlp", mlp, dists)
    return lifted_module()

# Inference
inference = SVI(model, guide, Adam({"lr": 0.01}), loss=Trace_ELBO())
