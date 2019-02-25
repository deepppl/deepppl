
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

stan_model_file = 'deepppl/tests/good/linear_regression.stan'
global_num_iterations = 3000
global_num_chains = 1


def nuts(model, **kwargs):
    nuts_kernel = mcmc.NUTS(model, adapt_step_size=True)
    return mcmc.MCMC(nuts_kernel, **kwargs)


def test_linear_regression():
    model = deepppl.DppplModel(
        model_file=stan_model_file)

    posterior = model.posterior(
        method=nuts,
        num_samples=2700,
        warmup_steps=300)

    # Add Data
    num_samples = 10
    X = np.arange(num_samples)
    y = np.arange(num_samples)
    data = {'N': num_samples,
            'x': X,
            'y': y}

    X = torch.Tensor(X)
    y = torch.Tensor(y)

    t1 = time.time()
    marginal = pyro.infer.EmpiricalMarginal(posterior.run(
        N=num_samples, x=X, y=y), sites=['alpha', 'beta', 'sigma'])
    samples_fstan = [marginal() for _ in range(1000)]
    stack_samples = torch.stack(samples_fstan)
    params = torch.mean(stack_samples, 0)
    t2 = time.time()

    time_taken = t2-t1
    print(params)
    alpha = float(params[0])
    beta = float(params[1])

    num_test_samples = 10
    X_test = np.arange(10, 10+num_test_samples)
    y_test = np.arange(10, 10+num_test_samples)
    y_predicted = alpha + beta * X_test

    y_pred, time_taken_pystan = compare_with_stan_output(data, X_test)

    print("mean_squared_error using deepstan:{} with time taken by inference:{}".format(
        mean_squared_error(y_test, y_predicted), time_taken))

    print("mean_squared_error using pystan:{} with time taken by inference:{}".format(
        mean_squared_error(y_test, y_pred), time_taken_pystan))


def compare_with_stan_output(data, X_test):
    stan_code = open(stan_model_file).read()
    t1 = time.time()

    # Compile and fit
    sm1 = pystan.StanModel(model_code=str(stan_code))
    fit_stan = sm1.sampling(
        data=data, iter=global_num_iterations, chains=global_num_chains, warmup=300)

    alpha = fit_stan.extract(permuted=True)['alpha'].mean()
    beta = fit_stan.extract(permuted=True)['beta'].mean()
    t2 = time.time()

    print(alpha, beta)

    y_predicted = alpha + beta * X_test
    return y_predicted, t2-t1


if __name__ == "__main__":
    test_linear_regression()
