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

def test_gaussian_log_density_inference():
    model = deepppl.DppplModel(model_file = 'deepppl/tests/good/gaussian_log_density.stan')

    posterior = model.posterior(
                method=nuts, 
                num_samples=3000, 
                warmup_steps=300)

    marginal = pyro.infer.EmpiricalMarginal(posterior.run(), sites='theta')

    series = pd.Series([marginal().item() for _ in range(3000)], name = r'$\theta$')
    assert np.abs(series.mean() - 1000) < 1
    assert np.abs(series.std() - 1.0) < 0.1 

