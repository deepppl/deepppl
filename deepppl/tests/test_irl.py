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
from deepppl.translation.exceptions import MissingPriorNetException, MissingGuideNetException,\
                                            MissingModelExeption, MissingGuideExeption,\
                                            ObserveOnGuideExeption, UnsupportedProperty,\
                                            UndeclaredParametersException, UndeclaredNetworkException,\
                                            InvalidSamplingException, UndeclaredVariableException,\
                                            UnknownDistributionException

import ast
import pytest

def code_to_normalized(code):
    return ast.dump(ast.parse(code))


def normalize_and_compare(src_file, target_file):
    with open(target_file) as f:
        target_code = f.read() 
    target = code_to_normalized(target_code)
    
    compiled = dpplc.stan2astpyFile(src_file)
    assert code_to_normalized(compiled) == target

def test_coin():
    filename = r'deepppl/tests/good/coin.stan'
    target_file = r'deepppl/tests/target_py/coin.py'
    normalize_and_compare(filename, target_file)


def test_operators():
    filename = r'deepppl/tests/good/operators.stan'
    target_file = r'deepppl/tests/target_py/operators.py'
    normalize_and_compare(filename, target_file)

def test_operators_expr():
    filename = r'deepppl/tests/good/operators-expr.stan'
    target_file = r'deepppl/tests/target_py/operators-expr.py'
    normalize_and_compare(filename, target_file)

def test_coin_vectorized():
    filename = r'deepppl/tests/good/coin_vectorized.stan'
    target_file = r'deepppl/tests/target_py/coin_vectorized.py'
    normalize_and_compare(filename, target_file)



@pytest.mark.xfail(strict=True)
def test_coin_reverted_lines():
    """Inside a `block`, stan semantics do not requires lines to be 
    ordered.
    """
    filename = r'deepppl/tests/good/coin_reverted.stan'
    target_file = r'deepppl/tests/target_py/coin_vectorized.py'
    normalize_and_compare(filename, target_file)


def test_coin_guide():
    filename = r'deepppl/tests/good/coin_guide.stan'
    target_file = r'deepppl/tests/target_py/coin_guide.py'
    normalize_and_compare(filename, target_file)


def test_lstm():
    filename = r'deepppl/tests/good/lstm.stan'
    target_file = r'deepppl/tests/target_py/lstm.py'
    normalize_and_compare(filename, target_file)



def test_coin_guide_missing_var():
    with pytest.raises(MissingGuideExeption):
        filename = r'deepppl/tests/good/coin_guide_missing_var.stan'
        dpplc.stan2astpyFile(filename)

def test_coin_guide_sample_obs():
    with pytest.raises(ObserveOnGuideExeption):
        filename = r'deepppl/tests/good/coin_guide_sample_obs.stan'
        dpplc.stan2astpyFile(filename)

def test_coin_guide_missing_model():
    with pytest.raises(MissingModelExeption):
        filename = r'deepppl/tests/good/coin_guide_missing_model.stan'
        dpplc.stan2astpyFile(filename)


def test_mlp_undeclared_parameters():
    with pytest.raises(UndeclaredParametersException):
        filename = r'deepppl/tests/good/mlp_undeclared_parameters.stan'
        dpplc.stan2astpyFile(filename)

def test_mlp_undeclared_network():
    with pytest.raises(UndeclaredNetworkException):
        filename = r'deepppl/tests/good/mlp_undeclared_net.stan'
        dpplc.stan2astpyFile(filename)

def test_mlp_missing_guide():
    with pytest.raises(MissingGuideNetException):
        filename = r'deepppl/tests/good/mlp_missing_guide.stan'
        dpplc.stan2astpyFile(filename)

def test_mlp_missing_model():
    with pytest.raises(MissingPriorNetException):
        filename = r'deepppl/tests/good/mlp_missing_model.stan'
        dpplc.stan2astpyFile(filename)

def test_coin_invalid_sampling():
    with pytest.raises(InvalidSamplingException):
        filename = r'deepppl/tests/good/coin_invalid_sampling.stan'
        dpplc.stan2astpyFile(filename)

def test_coin_unknown_identifier():
    with pytest.raises(UndeclaredVariableException):
        filename = r'deepppl/tests/good/coin_unknown_identifier.stan'
        dpplc.stan2astpyFile(filename)

def test_coin_unknown_distribution():
    with pytest.raises(UnknownDistributionException):
        filename = r'deepppl/tests/good/coin_unknown_distribution.stan'
        dpplc.stan2astpyFile(filename)

        

def test_mlp_unsupported_property():
    with pytest.raises(UnsupportedProperty):
        filename = r'deepppl/tests/good/mlp_unsupported_prop.stan'
        target_file = r'deepppl/tests/target_py/mlp.py'
        normalize_and_compare(filename, target_file)

def test_mlp_unsupported_property1():
    with pytest.raises(UnsupportedProperty):
        filename = r'deepppl/tests/good/mlp_unsupported_prop1.stan'
        target_file = r'deepppl/tests/target_py/mlp.py'
        normalize_and_compare(filename, target_file)

def test_mlp():
    filename = r'deepppl/tests/good/mlp.stan'
    target_file = r'deepppl/tests/target_py/mlp.py'
    normalize_and_compare(filename, target_file)


def test_vae():
    filename = r'deepppl/tests/good/vae.stan'
    target_file = r'deepppl/tests/target_py/vae.py'
    normalize_and_compare(filename, target_file)