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
from torch.distributions import constraints
from numpyro import distributions as np_dist
from numpyro.distributions import constraints as np_constraints
import jax.numpy as jnp
import torch

# Utils to be imported by PyroModel


def build_hooks(npyro=False):
    if npyro:
        d = np_dist
        const = np_constraints
        provider = jnp
    else:
        d = dist
        const = constraints
        provider = torch
        
    def categorical_logits(logits):
        return d.Categorical(logits=logits)


    def bernoulli_logit(logits):
        return d.Bernoulli(logits=logits)


    def binomial_logit(n, logits):
        return d.Binomial(n, logits=logits)


    def poisson_log(alpha):
        return d.Poisson(provider.exp(alpha))

    def new_zeros(x):
        if npyro:
            return jnp.zeros_like(x)
        else:
            return x.new_zeros(x.shape)

    class ImproperUniform(d.Normal):
        def __init__(self, shape=None):
            zeros = provider.zeros(shape) if shape else 0
            ones = provider.ones(shape) if shape else 1
            super(ImproperUniform, self).__init__(zeros, ones)

        def log_prob(self, x):
            return new_zeros(x)


    class LowerConstrainedImproperUniform(ImproperUniform):
        def __init__(self, lower_bound=0, shape=None):
            self.lower_bound = lower_bound
            super(LowerConstrainedImproperUniform, self).__init__(shape)
            self.support = const.greater_than(lower_bound)

        def sample(self, *args, **kwargs):
            s = d.Uniform(self.lower_bound, self.lower_bound + 2).sample(*args, **kwargs)
            return s


    class UpperConstrainedImproperUniform(ImproperUniform):
        def __init__(self, upper_bound=0.0, shape=None):
            self.upper_bound = upper_bound
            super(UpperConstrainedImproperUniform, self).__init__(shape)
            self.support = const.less_than(upper_bound)

        def sample(self, *args, **kwargs):
            s = d.Uniform(self.upper_bound - 2.0, self.upper_bound).sample(*args, **kwargs)
            return s

    def is_arrayed(x):
        cls = jnp.DeviceArray if npyro else torch.Tensor
        return isinstance(x, cls)
        
    def build_array(x):
        return jnp.array(x, dtype=float) if npyro else torch.tensor(x).float()

    def log(x):
        if not is_arrayed(x):
            x = build_array(x)
        return provider.log(x)


    def dot_self(x):
        return provider.dot(x, x)


    def log_sum_exp(x):
        f = jnp.logaddexp if npyro else torch.logsumexp
        return f(x, 0)


    def inv_logit(p):
        return provider.log(p / (1. - p))


    return {x.__name__: x for x in [
        bernoulli_logit,
        categorical_logits,
        binomial_logit,
        poisson_log,
        ImproperUniform,
        LowerConstrainedImproperUniform,
        UpperConstrainedImproperUniform,
        log,
        dot_self,
        log_sum_exp,
        inv_logit]

    }
