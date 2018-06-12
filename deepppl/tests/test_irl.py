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
# */

import dpplc
import ast

def code_to_normalized(code):
    return ast.dump(ast.parse(code))


def test_coin():
    filename = r'tests/good/coin.stan'
    target_code = """
import torch
import pyro
import pyro.distributions as dist


def model(x):
    theta = pyro.sample('theta', dist.Uniform(0, 1))
    for i in range(1, 10 + 1):
        pyro.sample('x' + '{}'.format(i - 1), dist.Bernoulli(theta), obs=x[
            i - 1])
""" 
    target = code_to_normalized(target_code)
    
    compiled = dpplc.stan2astpyFile(filename)
    assert code_to_normalized(compiled) == target

def test_coin_vectorized():
    filename = r'tests/good/coin_vectorized.stan'
    target_code = """
import torch
import pyro
import pyro.distributions as dist


def model(x):
    theta = pyro.sample('theta', dist.Uniform(0, 1))
    pyro.sample('x', dist.Bernoulli(theta), obs=x)
""" 
    target = code_to_normalized(target_code)
    
    compiled = dpplc.stan2astpyFile(filename)
    assert code_to_normalized(compiled) == target


