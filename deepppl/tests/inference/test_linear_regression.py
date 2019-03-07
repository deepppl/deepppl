import torch
import pyro
import pyro.distributions as dist
from pyro.infer import mcmc
import deepppl
import os
import numpy as np
import pandas as pd
import pystan
from sklearn.metrics import mean_squared_error
import time

stan_model_file = 'deepppl/tests/good/linear_regression.stan'
stan_array_model_file = 'deepppl/tests/good/linear_regression_array.stan'
global_num_iterations=3000
global_num_chains=1
global_warmup_steps = 300


def nuts(model, **kwargs):
    nuts_kernel = mcmc.NUTS(model, adapt_step_size=True)
    return mcmc.MCMC(nuts_kernel, **kwargs)


def test_linear_regression():
    model = deepppl.DppplModel(
        model_file=stan_array_model_file)

    posterior = model.posterior(
        method=nuts,
        num_samples=global_num_iterations - global_warmup_steps,
        warmup_steps=global_warmup_steps)

    # Add Data
    num_samples = 10
    X = np.arange(num_samples)
    # y = np.arange(num_samples)
    y = np.array([ dist.Normal(i, 0.1).sample() for i in range(num_samples)])
    data = {'N': num_samples,
            'x': X,
            'y': y}

    X = torch.Tensor(X)
    y = torch.Tensor(y)

    t1 = time.time()
    marginal = pyro.infer.EmpiricalMarginal(posterior.run(
        N=num_samples, x=X, y=y), sites=['alpha', 'beta', 'sigma'])
    samples_fstan = [marginal() for _ in range(global_num_iterations - global_warmup_steps)]
    stack_samples = torch.stack(samples_fstan)
    params = torch.mean(stack_samples, 0)
    t2 = time.time()
    print('Pyro: [ alpha, beta, sigma ].mean =', params)
    stack_samples = torch.stack(samples_fstan).numpy()

    params = ['alpha', 'beta', 'sigma']
    df = pd.DataFrame(stack_samples, columns=params)

    pystan_output, pystan_time, pystan_compilation_time = compare_with_stan_output(data)

    assert df.shape == pystan_output.shape
    from scipy.stats import entropy
    for column in params:
        #import pdb;pdb.set_trace()
        hist1 = np.histogram(df[column], bins = 10)
        hist2 = np.histogram(pystan_output[column], bins = hist1[1])
        kl = entropy(hist1[0]+1, hist2[0]+1)
        skl = kl + entropy(hist2[0]+1, hist1[0]+1)
        print('skl for column:{} is:{:.2f}'.format(column, skl))

    print("Time taken: deepstan:{:.2f}, pystan_compilation:{:.2f}, pystan:{:.2f}".format(t2-t1, pystan_compilation_time, pystan_time))


def compare_with_stan_output(data):
    stan_code = open(stan_model_file).read()
    t1 = time.time()

    # Compile and fit
    sm1 = pystan.StanModel(model_code=str(stan_code))
    t2 = time.time()
    pystan_compilation_time = t2-t1

    t1 = time.time()
    fit_stan = sm1.sampling(
        data=data, iter=global_num_iterations, chains=global_num_chains, warmup=300)

    alpha = fit_stan.extract(permuted=True)['alpha']
    beta = fit_stan.extract(permuted=True)['beta']
    sigma = fit_stan.extract(permuted=True)['sigma']

    alpha = np.reshape(alpha, (alpha.shape[0], 1))
    beta = np.reshape(beta, (beta.shape[0], 1))
    sigma = np.reshape(sigma, (sigma.shape[0], 1))

    marginal = np.concatenate((alpha, beta, sigma), axis = 1)
    params = ['alpha', 'beta', 'sigma']
    df = pd.DataFrame(marginal, columns=params)

    t2 = time.time()

    print('Stan: [ alpha, beta, sigma ].mean = [ {}, {}, {} ]'.format(alpha, beta, sigma))
    return df, t2-t1, pystan_compilation_time



if __name__ == "__main__":
    test_linear_regression()
