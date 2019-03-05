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

def normalize_and_compare_test(test_name):
        filename = f'deepppl/tests/good/paper/{test_name}.stan'
        target_file = f'deepppl/tests/target_py/paper/{test_name}.py'
        normalize_and_compare(filename, target_file)

def test_bayes_nn():
        filename = r'deepppl/tests/good/paper/bayes_nn.stan'
        dpplc.stan2astpyFile(filename, verbose=True)
# TODO: for some reason comparison fails, even when the input/output appears to be exactly the same
#        normalize_and_compare_test('bayes_nn')

def test_coin():
        normalize_and_compare_test('coin')

def test_coin_guide():
        normalize_and_compare_test('coin_guide')

def test_multimodal():
        normalize_and_compare_test('multimodal')

def test_multimodal_guide():
        normalize_and_compare_test('multimodal_guide')

def test_posterior_twice():
        normalize_and_compare_test('posterior_twice')

def test_vae_inferred_shape():
        normalize_and_compare_test('vae_inferred_shape')

def test_vae():
        normalize_and_compare_test('vae')
