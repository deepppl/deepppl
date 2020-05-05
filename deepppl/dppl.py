# /*
#  * Copyright 2018 IBM Corporation
#  *
#  * Licensed under the Apache License, Version 2.0 (the "License");
#  * you may not use this file except in compliance with the License.
#  * You may obtain a copy of the License at
#  *
#  * http://www.apache.org/licenses/LICENSE-2.0
#  *
#  * Unless required by applicable law or agreed to in writing, software
#  * distributed under the License is distributed on an "AS IS" BASIS,
#  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  * See the License for the specific language governing permissions and
#  * limitations under the License.
#  */

import pandas as pd
from collections import defaultdict
import pyro
from pyro.optim import Adam
import torch
import numpy as onp
import numpyro
import numpyro.distributions as dist
import jax.random as random
from jax import numpy as jnp

from torch.nn import functional as F
import sys
from . import dpplc
from .utils import utils
import inspect

class PyroModel(object):
    def __init__(self, model_code=None, model_file=None, **kwargs):
        self._py = self.compile(model_code=model_code, model_file=model_file)
        self._load_py()
        self._updateHooksAll(kwargs)

    def compile(self, **kwargs):
        return dpplc.do_compile(**kwargs)

    def _updateHooksAll(self, hooks):
        [self._updateHooks(f, hooks) for f in (self._model,
                                               self._guide,
                                               self._transformed_data,
                                               self._generated_quantities)]

    def _updateHooks(self, f, hooks):
        if f:
            f.__globals__.update(hooks)

    def _load_py(self):
        """Load the python object into `_model`"""
        locals_ = {}
        eval(self._py, globals(), locals_)
        self._model = locals_['model']
        self._updateHooks(self._model, locals_)
        self._guide = None
        self._transformed_data = None
        self._generated_quantities = None
        for k in locals_.keys():
            if k.startswith('guide_'):
                self._guide = locals_[k]
                self._updateHooks(self._model, locals_)
            if k.startswith('transformed_data'):
                self._transformed_data = locals_[k]
                self._updateHooks(self._model, locals_)
            if k.startswith('generated_quantities'):
                self._generated_quantities = locals_[k]
                self._updateHooks(self._model, locals_)
        self._loadBasicHooks()

    def _loadBasicHooks(self):
        hooks = {x.__name__: x for x in [
            torch.sqrt,
            torch.randn,
            torch.exp,
            torch.log,
            torch.zeros,
            torch.ones,
            F.softplus, 
            pyro.sample]}
        hooks['fabs'] = torch.abs
        self._updateHooksAll(hooks)
        self._updateUtilsHooks()
        
    def _updateUtilsHooks(self):
        self._updateHooksAll(utils.build_hooks())

    def mcmc(self, num_samples=10000, warmup_steps=1000, num_chains=1, kernel=None):
        if kernel is None:
            kernel = pyro.infer.NUTS(self._model, adapt_step_size=True)
        mcmc = pyro.infer.MCMC(kernel, num_samples, warmup_steps=warmup_steps, num_chains=num_chains) 
        return MCMCProxy(mcmc, False, self._generated_quantities, self._transformed_data)

    def svi(self, optimizer=None, loss=None, params={'lr': 0.0005, "betas": (0.90, 0.999)}):
        optimizer = optimizer if optimizer else Adam(params)
        loss = loss if loss is not None else pyro.infer.Trace_ELBO()
        svi = pyro.infer.SVI(self._model, self._guide, optimizer, loss)
        return SVIProxy(svi)

class NumPyroModel(PyroModel):
    def _loadBasicHooks(self):
        hooks = {x.__name__: x for x in [
            jnp.sqrt,
            onp.random.randn,
            jnp.exp,
            jnp.log,
            jnp.zeros,
            jnp.ones,
            numpyro.sample]}
        hooks['softplus'] = lambda x: jnp.logaddexp(x, 0.)
        hooks['fabs'] = torch.abs
        self._updateHooksAll(hooks)
        self._updateUtilsHooks()
        
    def compile(self, **kwargs):
        config = dpplc.Config(numpyro=True)
        return super(NumPyroModel, self).compile(config=config, **kwargs)
    
    def mcmc(self, num_samples=10000, warmup_steps=1000, num_chains=1, kernel=None):
        if kernel is None:
            kernel = numpyro.infer.NUTS(self._model, adapt_step_size=True) 
        mcmc = numpyro.infer.MCMC(kernel, warmup_steps, num_samples, num_chains)
        return MCMCProxy(mcmc, True, self._generated_quantities, self._transformed_data)
    
    def _updateUtilsHooks(self):
        self._updateHooksAll(utils.build_hooks(npyro=True))
 

class MCMCProxy():
    def __init__(self, mcmc, numpyro=False, generated_quantities=None, transformed_data=None):
        self.mcmc = mcmc
        self.transformed_data = transformed_data
        self.generated_quantities = generated_quantities
        self.numpyro = numpyro
        self.data = None
        if numpyro:
            self.rng_key, _ = random.split(random.PRNGKey(0))
        
    def run(self, *args, **kwargs):
        self.data = kwargs
        if self.transformed_data:
            self.data['transformed_data'] = self.transformed_data(**self.data)
        if self.numpyro:
            kwargs = {k: _convert_to_np(v) for k,v in self.data.items()}
            self.mcmc.run(self.rng_key, *args, **kwargs) 
        else:
            kwargs = {k: _convert_to_tensor(v) for k,v in self.data.items()}
            self.mcmc.run(*args, **kwargs)
            
    def sample_model(self):
        # Pyro removes warmup steps from samples
        if self.numpyro:
            warmup = self.mcmc.num_warmup
            samples = self.mcmc.get_samples()
        else:
            warmup = self.mcmc.warmup_steps
            samples = self.mcmc.get_samples()
        return {x: samples[x][warmup:] for x in samples}
        
    def sample_generated(self, samples):
        kwargs = self.data
        res = defaultdict(list)
        num_samples = len(list(samples.values())[0])
        for i in range(num_samples):
            kwargs['parameters'] = {x: samples[x][i] for x in samples}
            d = self.generated_quantities(**kwargs)
            for k, v in d.items():
                res[k].append(_convert_to_np(v))
        return res
     
    def get_samples(self):
        samples = self.sample_model()
        if self.generated_quantities:
            gen = self.sample_generated(samples)
            samples.update(gen)
        return {k: _convert_to_np(v) for k, v in samples.items()}
            
        
        
           
            
        
class SVIProxy(object):
    def __init__(self, svi):
        self.svi = svi

    def posterior(self, n):
        signature = inspect.signature(self.svi.guide)
        args = [None for i in range(len(signature.parameters))]
        return [self.svi.guide(*args) for _ in range(n)]

    def step(self, *args):
        return self.svi.step(*args)


def _convert_to_np(value):
    if type(value) == torch.Tensor:
        return value.cpu().numpy()
    if isinstance(value, list):
        return onp.array(value)
    else:
        return value
        
def _convert_to_tensor(value):
    if isinstance(value, (list, onp.ndarray)):
        return torch.Tensor(value)
    elif isinstance(value, dict):
        return {k: _convert_to_tensor(v) for k, v in value.items()}
    else:
        return value
