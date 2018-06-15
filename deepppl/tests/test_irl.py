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
import pytest

def code_to_normalized(code):
    return ast.dump(ast.parse(code))


def test_coin():
    filename = r'tests/good/coin.stan'
    target_file = r'tests/target_py/coin.py'
    with open(target_file) as f:
        target_code = f.read() 
    target = code_to_normalized(target_code)
    
    compiled = dpplc.stan2astpyFile(filename)
    assert code_to_normalized(compiled) == target

def test_operators():
    filename = r'tests/good/operators.stan'
    target_file = r'tests/target_py/operators.py'
    with open(target_file) as f:
        target_code = f.read() 
    target = code_to_normalized(target_code)
    
    compiled = dpplc.stan2astpyFile(filename)
    assert code_to_normalized(compiled) == target

def test_operators_expr():
    filename = r'tests/good/operators-expr.stan'
    target_file = r'tests/target_py/operators-expr.py'
    with open(target_file) as f:
        target_code = f.read() 
    target = code_to_normalized(target_code)
    
    compiled = dpplc.stan2astpyFile(filename)
    assert code_to_normalized(compiled) == target

def test_coin_vectorized():
    filename = r'tests/good/coin_vectorized.stan'
    target_file = r'tests/target_py/coin_vectorized.py'
    with open(target_file) as f:
        target_code = f.read() 
    target = code_to_normalized(target_code)
    
    compiled = dpplc.stan2astpyFile(filename)
    assert code_to_normalized(compiled) == target



@pytest.mark.xfail(strict=True)
def test_coin_reverted_lines():
    """Inside a `block`, stan semantics do not requires lines to be 
    ordered.
    """
    filename = r'tests/good/coin_reverted.stan'
    target_file = r'tests/target_py/coin_vectorized.py'
    with open(target_file) as f:
        target_code = f.read() 
    target = code_to_normalized(target_code)
    
    compiled = dpplc.stan2astpyFile(filename)
    assert code_to_normalized(compiled) == target


def test_mlp():
    filename = r'tests/good/mlp.stan'
    target_file = r'tests/target_py/coin_vectorized.py'
    with open(target_file) as f:
        target_code = f.read() 
    target = code_to_normalized(target_code)
    
    compiled = dpplc.stan2astpyFile(filename)
    assert code_to_normalized(compiled) == target

