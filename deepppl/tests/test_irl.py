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

    compiled = dpplc.stan2astpyFile(src_file)
    assert code_to_normalized(compiled) == target


def test_coin():
    filename = r'deepppl/tests/good/coin.stan'
    target_file = r'deepppl/tests/target_py/coin.py'
    normalize_and_compare(filename, target_file)


def test_gaussian():
    filename = r'deepppl/tests/good/gaussian.stan'
    target_file = r'deepppl/tests/target_py/gaussian.py'
    normalize_and_compare(filename, target_file)


def test_gaussian_log_density():
    filename = r'deepppl/tests/good/gaussian_log_density.stan'
    target_file = r'deepppl/tests/target_py/gaussian_log_density.py'
    normalize_and_compare(filename, target_file)


def test_double_gaussian():
    filename = r'deepppl/tests/good/double_gaussian.stan'
    target_file = r'deepppl/tests/target_py/double_gaussian.py'
    normalize_and_compare(filename, target_file)


def test_log_normal():
    filename = r'deepppl/tests/good/log_normal.stan'
    target_file = r'deepppl/tests/target_py/log_normal.py'
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


def test_coin_transormed_data():
    filename = r'deepppl/tests/good/coin_transformed_data.stan'
    target_file = r'deepppl/tests/target_py/coin_transformed_data.py'
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


def test_coin_guide_init():
    filename = r'deepppl/tests/good/coin_guide_init.stan'
    target_file = r'deepppl/tests/target_py/coin_guide_init.py'
    normalize_and_compare(filename, target_file)


def test_simple_init():
    filename = r'deepppl/tests/good/simple_init.stan'
    target_file = r'deepppl/tests/target_py/simple_init.py'
    normalize_and_compare(filename, target_file)


def test_lstm():
    filename = r'deepppl/tests/good/lstm.stan'
    target_file = r'deepppl/tests/target_py/lstm.py'
    normalize_and_compare(filename, target_file)


def test_coin_guide_missing_var():
    with pytest.raises(MissingGuideException):
        filename = r'deepppl/tests/good/coin_guide_missing_var.stan'
        dpplc.stan2astpyFile(filename)


def test_coin_guide_sample_obs():
    with pytest.raises(ObserveOnGuideException):
        filename = r'deepppl/tests/good/coin_guide_sample_obs.stan'
        dpplc.stan2astpyFile(filename)


def test_coin_guide_missing_model():
    "Implicit prior allows to write missing model."
    with not_raises(MissingModelException):
        filename = r'deepppl/tests/good/coin_guide_missing_model.stan'
        dpplc.stan2astpyFile(filename)


def test_mlp_undeclared_parameters():
    with pytest.raises(UndeclaredParametersException):
        filename = r'deepppl/tests/good/mlp_undeclared_parameters.stan'
        dpplc.stan2astpyFile(filename)


def test_mlp_undeclared_network1():
    with pytest.raises(UndeclaredNetworkException):
        filename = r'deepppl/tests/good/mlp_undeclared_net1.stan'
        dpplc.stan2astpyFile(filename)


def test_mlp_undeclared_network2():
    with pytest.raises(UndeclaredNetworkException):
        filename = r'deepppl/tests/good/mlp_undeclared_net2.stan'
        dpplc.stan2astpyFile(filename)


def test_mlp_missing_guide():
    with pytest.raises(MissingGuideNetException):
        filename = r'deepppl/tests/good/mlp_missing_guide.stan'
        dpplc.stan2astpyFile(filename)


def test_mlp_missing_model():
    with not_raises(MissingPriorNetException):
        filename = r'deepppl/tests/good/mlp_missing_model.stan'
        dpplc.stan2astpyFile(filename)


def test_mlp_incorrect_shape1():
    with pytest.raises(IncompatibleShapes):
        filename = r'deepppl/tests/good/mlp_incorrect_shape1.stan'
        dpplc.stan2astpyFile(filename)


def test_mlp_incorrect_shape2():
    with pytest.raises(IncompatibleShapes):
        filename = r'deepppl/tests/good/mlp_incorrect_shape2.stan'
        dpplc.stan2astpyFile(filename)


def test_mlp_incorrect_shape3():
    with pytest.raises(IncompatibleShapes):
        filename = r'deepppl/tests/good/mlp_incorrect_shape3.stan'
        dpplc.stan2astpyFile(filename)


def test_mlp_incorrect_shape4():
    with pytest.raises(IncompatibleShapes):
        filename = r'deepppl/tests/good/mlp_incorrect_shape4.stan'
        dpplc.stan2astpyFile(filename)


def test_coin_invalid_sampling():
    with not_raises(InvalidSamplingException):
        filename = r'deepppl/tests/good/coin_invalid_sampling.stan'
        dpplc.stan2astpyFile(filename)


def test_coin_invalid_sampling2():
    with not_raises(NonRandomSamplingException):
        filename = r'deepppl/tests/good/coin_invalid_sampling2.stan'
        dpplc.stan2astpyFile(filename)


def test_coin_unknown_identifier():
    with pytest.raises(UndeclaredVariableException):
        filename = r'deepppl/tests/good/coin_unknown_identifier.stan'
        dpplc.stan2astpyFile(filename)


def test_coin_unknown_distribution():
    with pytest.raises(UnknownDistributionException):
        filename = r'deepppl/tests/good/coin_unknown_distribution.stan'
        dpplc.stan2astpyFile(filename)


def test_coin_already_declared():
    with pytest.raises(AlreadyDeclaredException):
        filename = r'deepppl/tests/good/coin_already_declared_var.stan'
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


def test_mlp_default_init():
    filename = r'deepppl/tests/good/mlp_default_init.stan'
    target_file = r'deepppl/tests/target_py/mlp_init.py'
    normalize_and_compare(filename, target_file)


def test_vae():
    filename = r'deepppl/tests/good/vae.stan'
    target_file = r'deepppl/tests/target_py/vae.py'
    normalize_and_compare(filename, target_file)


def test_linear_regression():
    filename = r'deepppl/tests/good/linear_regression.stan'
    target_file = r'deepppl/tests/target_py/linear_regression.py'
    normalize_and_compare(filename, target_file)


def test_kmeans():
    filename = r'deepppl/tests/good/kmeans.stan'
    target_file = r'deepppl/tests/target_py/kmeans.py'
    normalize_and_compare(filename, target_file)


def test_schools():
    filename = r'deepppl/tests/good/schools.stan'
    target_file = r'deepppl/tests/target_py/schools.py'
    normalize_and_compare(filename, target_file)


def test_gaussian_process():
    filename = r'deepppl/tests/good/gaussian_process.stan'
    target_file = r'deepppl/tests/target_py/gaussian_process.py'
    normalize_and_compare(filename, target_file)


def test_missing_data():
    filename = r'deepppl/tests/good/missing_data.stan'
    target_file = r'deepppl/tests/target_py/missing_data.py'
    normalize_and_compare(filename, target_file)


def test_regression_matrix():
    filename = r'deepppl/tests/good/regression_matrix.stan'
    target_file = r'deepppl/tests/target_py/regression_matrix.py'
    normalize_and_compare(filename, target_file)


def test_logistic():
    filename = r'deepppl/tests/good/logistic.stan'
    target_file = r'deepppl/tests/target_py/logistic.py'
    normalize_and_compare(filename, target_file)

# def test_row_vector_expr_terms():
#     filename = r'deepppl/tests/good/row_vector_expr_terms.stan'
#     target_file = r'deepppl/tests/target_py/row_vector_expr_terms.py'
#     normalize_and_compare(filename, target_file)


def test_gradient_warn():
    filename = r'deepppl/tests/good/gradient_warn.stan'
    target_file = r'deepppl/tests/target_py/gradient_warn.py'
    normalize_and_compare(filename, target_file)


def test_squared_error():
    filename = r'deepppl/tests/good/squared_error.stan'
    target_file = r'deepppl/tests/target_py/squared_error.py'
    normalize_and_compare(filename, target_file)

# def test_validate_arr_expr_primitives():
#     filename = r'deepppl/tests/good/validate_arr_expr_primitives.stan'
#     target_file = r'deepppl/tests/target_py/validate_arr_expr_primitives.py'
#     normalize_and_compare(filename, target_file)


def test_vectorized_probability():
    filename = r'deepppl/tests/good/vectorized_probability.stan'
    target_file = r'deepppl/tests/target_py/vectorized_probability.py'
    normalize_and_compare(filename, target_file)


def test_lda():
    filename = r'deepppl/tests/good/lda.stan'
    target_file = r'deepppl/tests/target_py/lda.py'
    normalize_and_compare(filename, target_file)


def test_neal_funnel():
    filename = r'deepppl/tests/good/neal_funnel.stan'
    target_file = r'deepppl/tests/target_py/neal_funnel.py'
    normalize_and_compare(filename, target_file)


def test_cockroaches():
    filename = r'deepppl/tests/good/cockroaches.stan'
    target_file = r'deepppl/tests/target_py/cockroaches.py'
    normalize_and_compare(filename, target_file)
