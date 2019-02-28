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

from deepppl import dpplc
from deepppl.translation.exceptions import *
from contextlib import contextmanager

import ast
import pytest


def code_to_normalized(code):
    return ast.dump(ast.parse(code))


from contextlib import contextmanager


@contextmanager
def not_raises(exception):
    try:
        yield
    except exception:
        raise pytest.fail("DID RAISE {0}".format(exception))


def normalize_and_compare(src_file, target_file):
    with open(target_file) as f:
        target_code = f.read()
    target = code_to_normalized(target_code)
    
    compiled = dpplc.stan2astpyFile(src_file, verbose=True)
    assert code_to_normalized(compiled) == target


# def test_coin():
#     filename = r'deepppl/tests/good/coin.stan'
#     target_file = r'deepppl/tests/target_py/coin.py'
#     normalize_and_compare(filename, target_file)


def test_bayes_nn():
        filename = r'deepppl/tests/paper/bayes_nn.stan'
        dpplc.stan2astpyFile(filename)

def test_biased_coin():
        filename = r'deepppl/tests/paper/biased_coin.stan'
        dpplc.stan2astpyFile(filename)

def test_coin():
        filename = r'deepppl/tests/paper/coin.stan'
        dpplc.stan2astpyFile(filename)

def test_multimodel():
        filename = r'deepppl/tests/paper/multimodal.stan'
        dpplc.stan2astpyFile(filename)

def test_multimodel_guide():
        filename = r'deepppl/tests/paper/multimodal_guide.stan'
        dpplc.stan2astpyFile(filename)

def test_posterior_twice():
        filename = r'deepppl/tests/paper/posterior_twice.stan'
        dpplc.stan2astpyFile(filename)

def test_vae():
        filename = r'deepppl/tests/paper/vae.stan'
        dpplc.stan2astpyFile(filename)
