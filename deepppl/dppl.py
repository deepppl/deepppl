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

import dpplc
import pyro
from pyro import infer
from pyro.optim import Adam
import sys

class DppplModel(object):
    def __init__(self, model_code = None, model_file = None):
        self._py = dpplc.do_compile(model_code = model_code, model_file = model_file)
        self._load_py()

    def _load_py(self):
        """Load the python object into `_model`"""
        locals_ = {}
        eval(self._py, globals(), locals_)
        self._model = locals_['model']
        self._model.__globals__.update(locals_)
        self._guide = None
        for k in locals_.keys():
            if k.startswith('guide_'):
                self._guide = locals_[k]
                self._model.__globals__.update(locals_)
                break
        
    def posterior(self, num_samples=3000, method=infer.Importance):
        return method(self._model, num_samples=3000)

    def svi(self, optimizer = None, loss = infer.Trace_ELBO(), params = {'lr' : 0.0005}):
        optimizer = optimizer if optimizer else Adam(params)
        svi = infer.SVI(self._model, self._guide, optimizer, loss)
        return svi
