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


import pyro
from pyro import infer
from pyro.optim import Adam
import torch
from torch.nn import functional as F
import sys
from . import dpplc
from .utils import utils
import inspect


class DppplModel(object):
    def __init__(self, model_code = None, model_file = None, **kwargs):
        self._py = dpplc.do_compile(model_code = model_code, model_file = model_file)
        self._load_py()
        self._updateHooksAll(kwargs)


    def _updateHooksAll(self, hooks):
        [self._updateHooks(f, hooks) for f in (self._model, self._guide)]

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
        for k in locals_.keys():
            if k.startswith('guide_'):
                self._guide = locals_[k]
                self._updateHooks(self._model, locals_)
                break
        self._loadBasicHooks()

    def _loadBasicHooks(self):
        hooks = { x.__name__ : x for x in [
                        torch.randn,
                        torch.exp,
                        torch.log,
                        torch.zeros,
                        torch.ones, 
                        F.softplus]}
        hooks['fabs'] = torch.abs
        self._updateHooksAll(hooks)
        self._updateHooksAll(utils.hooks)
        
    def posterior(self, num_samples=3000, method=infer.Importance, **kwargs):
        return method(self._model, num_samples=3000, **kwargs)

    def svi(self, optimizer = None, loss = None, params = {'lr' : 0.0005, "betas": (0.90, 0.999)}):
        optimizer = optimizer if optimizer else Adam(params)
        loss = loss if loss is not None else infer.Trace_ELBO()
        svi = infer.SVI(self._model, self._guide, optimizer, loss)
        return SVIProxy(svi)


class SVIProxy(object):
    def __init__(self, svi):
        self.svi = svi

    def posterior(self, n):
        signature = inspect.signature(self.svi.guide)
        args = [None for i in range(len(signature.parameters))]
        return [self.svi.guide(*args) for _ in range(n)]

    def step(self, *args):
        return self.svi.step(*args)

