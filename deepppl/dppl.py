# /*
#  * Copyright 2018 
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

from collections import defaultdict
from types import FunctionType
import inspect
import builtins

import torch
import jax
import numpy as onp
from jax import numpy as jnp

import pyro
import numpyro

from . import dpplc
from .utils import utils



class PyroModel(object):
    def __init__(self, model_code=None, model_file=None, **kwargs):
        self._scope = kwargs
        self._py = self.compile(model_code=model_code, model_file=model_file)
        self._load_py()

    def compile(self, **kwargs):
        return dpplc.do_compile(**kwargs)

    def _load_py(self):
        """Load the python object into `_model`"""
        locals_ = {}
        eval(self._py, globals(), locals_)
        self._scope.update(locals_)
        self._guide = None
        self._prior = None
        self._transformed_data = None
        self._generated_quantities = None
        for k in locals_.keys():
            if k.startswith('guide_'):
                self._guide = self._stan_scoped(k)
            if k.startswith('prior'):
                self._scope[k] = self._stan_scoped(k)
            if k.startswith('transformed_data'):
                self._transformed_data = self._np_scoped(k)
            if k.startswith('generated_quantities'):
                self._generated_quantities = self._np_scoped(k)
        self._model = self._stan_scoped('model')

    def _stan_scoped(self, name):
        scoped = {k: v for k, v in builtins.__dict__.items()}
        scoped.update(utils.build_hooks())
        scoped['dist'] = pyro.distributions
        scoped['constraints'] = torch.distributions.constraints
        scoped.update({x.__name__: x
                       for x in [torch.tensor,
                                 torch.sqrt,
                                 torch.rand,
                                 torch.randn,
                                 torch.exp,
                                 torch.log,
                                 torch.zeros,
                                 torch.ones,
                                 torch.nn.functional.softplus,
                                 pyro.sample]})
        scoped['fabs'] = torch.abs
        scoped.update(self._scope)
        f = self._scope[name]
        scoped[name] = FunctionType(f.__code__, scoped, name)
        return scoped[name]

    def _np_scoped(self, name):
        f = self._scope[name]
        scoped = {k: v for k, v in builtins.__dict__.items()}
        scoped.update({x.__name__: x
                       for x in [onp.sqrt,
                                 onp.random.randn,
                                 onp.exp,
                                 onp.log,
                                 onp.zeros,
                                 onp.ones]})
        scoped['dot_self'] = lambda x: onp.dot(x, x)
        f = self._scope[name]
        scoped[name] = FunctionType(f.__code__, scoped, name)
        return scoped[name]

    def mcmc(self, num_samples=10000, warmup_steps=1000, num_chains=1, thin=1, kernel=None):
        if kernel is None:
            kernel = pyro.infer.NUTS(self._model, adapt_step_size=True)
        mcmc = pyro.infer.MCMC(
            kernel, num_samples - warmup_steps, warmup_steps=warmup_steps, num_chains=num_chains)
        return MCMCProxy(mcmc, False, self._generated_quantities, self._transformed_data, thin)

    def svi(self, optimizer=None, loss=None, params={'lr': 0.0005, "betas": (0.90, 0.999)}):
        optimizer = optimizer if optimizer else pyro.optim.Adam(params)
        loss = loss if loss is not None else pyro.infer.Trace_ELBO()
        svi = pyro.infer.SVI(self._model, self._guide, optimizer, loss)
        return SVIProxy(svi, self._generated_quantities, self._transformed_data)


class NumPyroModel(PyroModel):
    def _stan_scoped(self, name):
        f = self._scope[name]
        scoped = {k: v for k, v in builtins.__dict__.items()}
        scoped.update(utils.build_hooks(npyro=True))
        scoped['dist'] = numpyro.distributions
        scoped['constraints'] = numpyro.distributions.constraints
        scoped.update({x.__name__: x
                       for x in [jnp.sqrt,
                                 jnp.exp,
                                 jnp.log,
                                 jnp.zeros,
                                 jnp.ones,
                                 numpyro.sample]})
        scoped['softplus'] = lambda x: jnp.logaddexp(x, 0.)
        scoped['fabs'] = jnp.abs
        scoped.update(self._scope)
        f = self._scope[name]
        scoped[name] = FunctionType(f.__code__, scoped, name)
        return scoped[name]

    def compile(self, **kwargs):
        config = dpplc.Config(numpyro=True)
        return super(NumPyroModel, self).compile(config=config, **kwargs)

    def mcmc(self, num_samples=10000, warmup_steps=1000, num_chains=1, thin=1, kernel=None):
        if kernel is None:
            kernel = numpyro.infer.NUTS(self._model, adapt_step_size=True)
        mcmc = numpyro.infer.MCMC(
            kernel, warmup_steps, num_samples - warmup_steps, num_chains=num_chains)
        return MCMCProxy(mcmc, True, self._generated_quantities, self._transformed_data, thin)


class MCMCProxy():
    def __init__(self, mcmc, numpyro=False, generated_quantities=None, transformed_data=None, thin=1):
        self.mcmc = mcmc
        self.transformed_data = transformed_data
        self.generated_quantities = generated_quantities
        self.thin = thin
        self.numpyro = numpyro
        self.args = []
        self.kwargs = {}
        if numpyro:
            self.rng_key, _ = jax.random.split(jax.random.PRNGKey(0))

    def run(self, *args, **kwargs):
        self.args = [_convert_to_np(v) for v in args]
        self.kwargs = {k: _convert_to_np(v) for k, v in kwargs.items()}
        if self.transformed_data:
            self.kwargs['transformed_data'] = self.transformed_data(
                *self.args, **self.kwargs)
        if self.numpyro:
            self.mcmc.run(self.rng_key, *self.args, **self.kwargs)
        else:
            args = [_convert_to_tensor(v) for v in self.args]
            kwargs = {k: _convert_to_tensor(v) for k, v in self.kwargs.items()}
            self.mcmc.run(*args, **kwargs)

    def sample_model(self):
        if self.numpyro:
            samples = self.mcmc.get_samples()
        else:
            samples = self.mcmc.get_samples()
        return {x: samples[x][::self.thin] for x in samples}

    def sample_generated(self, samples):
        kwargs = self.kwargs
        res = defaultdict(list)
        num_samples = len(list(samples.values())[0])
        for i in range(num_samples):
            kwargs['parameters'] = {x: samples[x][i] for x in samples}
            d = self.generated_quantities(*self.args, **kwargs)
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
    def __init__(self, svi, generated_quantities=None, transformed_data=None):
        self.svi = svi
        self.transformed_data = transformed_data
        self.generated_quantities = generated_quantities
        self.args = []
        self.kwargs = {}

    def posterior(self, n):
        signature = inspect.signature(self.svi.guide)
        args = [None for i in range(len(signature.parameters))]
        return [self.svi.guide(*args) for _ in range(n)]

    def step(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        if self.transformed_data:
            self.kwargs['transformed_data'] = self.transformed_data(
                **self.args, **self.kwargs)
        args = [_convert_to_tensor(v) for v in self.args]
        kwargs = {k: _convert_to_tensor(v) for k, v in self.kwargs.items()}
        return self.svi.step(*args, **kwargs)


def _convert_to_np(value):
    if type(value) == torch.Tensor:
        return value.cpu().numpy()
    if isinstance(value, (list, jnp.ndarray)):
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
