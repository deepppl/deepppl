
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import mcmc
import deepppl
import os
import numpy as np
import pystan
from sklearn.metrics import mean_squared_error
import time

from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.contrib.autoguide import AutoDiagonalNormal


def nuts(model, **kwargs):
    nuts_kernel = mcmc.NUTS(model, adapt_step_size=True)
    return mcmc.MCMC(nuts_kernel, **kwargs)


import torch
from torch import tensor, rand
import pyro
import torch.distributions.constraints as constraints
import pyro.distributions as dist
from deepppl.utils.utils import ImproperUniform, LowerConstrainedImproperUniform
import sys


# def model2(N, x, y):
#     alpha = pyro.sample('alpha', dist.Normal(0, 1))
#     beta = pyro.sample('beta', dist.Normal(0, 1))
#     sigma = pyro.sample('sigma', dist.Uniform(0, 10))
#     # with pyro.iarange('data', N):
#     pyro.sample('y' + '1', dist.Normal(alpha + beta * x, sigma), obs=y)


# def guide(N, x, y):
#     a_loc = pyro.param('a_loc', torch.tensor(0.))
#     a_scale = pyro.param('a_scale', torch.tensor(
#         1.), constraint=constraints.positive)
#     b_loc = pyro.param('b_loc', torch.tensor(0.))
#     b_scale = pyro.param('b_scale', torch.tensor(
#         1.), constraint=constraints.positive)
#     s_loc = pyro.param('s_loc', torch.tensor(
#         1.), constraint=constraints.positive)
#     alpha = pyro.sample('alpha', dist.Normal(a_loc, a_scale))
#     beta = pyro.sample('beta', dist.Normal(b_loc, b_scale))
#     sigma = pyro.sample('sigma', dist.Normal(s_loc, torch.tensor(0.05)))


def test_linear_regression():
    model = deepppl.DppplModel(
<< << << < HEAD
        model_file='deepppl/tests/good/linear_regression.stan')

    # model._model = model2

    # model = dmodel._model
    # guide = AutoDiagonalNormal(model)

    # Add Data
    num_samples = 50


== == == =
        model_file = stan_model_file)
    posterior=model.posterior(
        method = nuts,
        num_samples = 2700,
        warmup_steps = 300)

    # Add Data
    num_samples=10
>> >>>> > master
    X=np.arange(num_samples)
    y=np.arange(num_samples)
    data={'N': num_samples,
            'x': X,
            'y': y}

    X = torch.Tensor(X)
    y = torch.Tensor(y)

<< << << < HEAD
    # optim = Adam({"lr": 0.03})
    # svi = SVI(model, guide, optim, loss=Trace_ELBO(), num_samples=1000)

    # for step in range(5000):
    #     svi.step(num_samples, X, y)
    #     if step % 100 == 0:
    #         print('.', end='')
    #         sys.stdout.flush()

    posterior = model.posterior(
        method = nuts,
        num_samples = 3000,
        warmup_steps = 300)

    marginal=pyro.infer.EmpiricalMarginal(
        posterior.run(num_samples, X, y),
        sites = ['alpha', 'beta', 'sigma'])
    samples_fstan=[marginal() for _ in range(30)]
    stack_samples=torch.stack(samples_fstan)
    params=torch.mean(stack_samples, 0)
    # alpha = pyro.param("a_loc").item()
    # beta = pyro.param("b_loc").item()
    # sigma = pyro.param("s_loc").item()

    # print(alpha, beta, sigma)

    # for name, value in pyro.get_param_store():
    #     print(name, pyro.param(name), value)

    # print(params)
== == == =
    t1=time.time()
    marginal=pyro.infer.EmpiricalMarginal(posterior.run(
        N=num_samples, x=X, y=y), sites = ['alpha', 'beta', 'sigma'])
    samples_fstan=[marginal() for _ in range(1000)]
    stack_samples=torch.stack(samples_fstan)
    params=torch.mean(stack_samples, 0)
    t2=time.time()

    time_taken=t2-t1
    print(params)
    alpha=float(params[0])
    beta=float(params[1])

    num_test_samples=100
    X_test=np.arange(10, 10+num_test_samples)
    y_test=np.arange(10, 10+num_test_samples)
    y_predicted=alpha + beta * X_test

    y_pred, time_taken_pystan=compare_with_stan_output(data, X_test)

    print("mean_squared_error using deepstan:{} with time taken by inference:{}".format(
        mean_squared_error(y_test, y_predicted), time_taken))

    print("mean_squared_error using pystan:{} with time taken by inference:{}".format(
        mean_squared_error(y_test, y_pred), time_taken_pystan))


def compare_with_stan_output(data, X_test):
    stan_code=open(stan_model_file).read()
    t1=time.time()

    # Compile and fit
    sm1=pystan.StanModel(model_code = str(stan_code))
    fit_stan=sm1.sampling(data = data, iter = global_num_iterations,
                          chains = global_num_chains, warmup = 300)

    alpha=fit_stan.extract(permuted = True)['alpha'].mean()
    beta=fit_stan.extract(permuted = True)['beta'].mean()
    t2=time.time()

    print(alpha, beta)
>> >>>> > master

    y_predicted=alpha + beta * X_test
    return y_predicted, t2-t1

if __name__ == "__main__":
    test_linear_regression()
