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
import sys

class DppplModel(object):
    def __init__(self, model_code = None, model_file = None):
        self._py = dpplc.do_compile(model_code = model_code, model_file = model_file)
        self._load_py()

    def _load_py(self):
        """Load the python object into `_model`"""
        locals_ = {}
        eval(self._py, globals(), locals_)
        _model = locals_['model']
        for name in locals_:
            value = locals_[name]
            ## Check for module objects
            if type(value) == type(sys):
                _model.__globals__[name] = value

        self._model = _model
        
    def posterior(self, num_samples=3000, method=pyro.infer.Importance):
        return pyro.infer.Importance(self._model, num_samples=3000)
