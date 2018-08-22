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

from pyro import distributions as dist
import torch

## Utils to be imported by DppplModel

def CategoricalLogits(logits):
    return dist.Categorical(logits=logits)

class ImproperUniform(dist.Normal):
    def __init__(self, shape = None):
        zeros = torch.zeros(shape) if shape else 0
        ones = torch.ones(shape) if shape else 1
        super(ImproperUniform, self).__init__(zeros, ones)
        
    def log_prob(self, x):
        return x.new_zeros(x.shape)

hooks = {
    CategoricalLogits.__name__ : CategoricalLogits,
    ImproperUniform.__name__ : ImproperUniform
}