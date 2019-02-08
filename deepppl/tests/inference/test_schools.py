import torch
import torch
from torch import nn
from torch.nn import functional as F
from torch import tensor
import pyro
from pyro import distributions as dist
import numpy as np
from pyro.infer import mcmc

import deepppl
import os
import pandas as pd

def nuts(model, **kwargs):
    nuts_kernel = mcmc.NUTS(model, adapt_step_size = True)
    return mcmc.MCMC(nuts_kernel, **kwargs)

def test_schools():
    model = deepppl.DppplModel(model_file = 'deepppl/tests/good/schools.stan')

    J = 8
    y = torch.from_numpy(np.array([28,  8, -3,  7, -1,  1, 18, 12], dtype=np.float))
    sigma = torch.from_numpy(np.array([15, 10, 16, 11, 9, 11, 10, 18], dtype=np.float))

    posterior = model.posterior(
                method=nuts, 
                num_samples=3000, 
                warmup_steps=300)

    marginal = pyro.infer.EmpiricalMarginal(posterior.run(J, y, sigma), sites=['mu','tau','eta'])

    series = pd.Series([marginal().item() for _ in range(3000)], name = r'$mu$')
    print(series)
    # assert np.abs(series.mean() - 1000) < 1
    # assert np.abs(series.std() - 1.0) < 0.1 

if __name__ == "__main__":
    test_schools()