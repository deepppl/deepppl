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
from deepppl.translation.stype import Type_

from contextlib import contextmanager
import pytest


def code_to_normalized(code):
    return ast.dump(ast.parse(code))

from contextlib import contextmanager

@contextmanager
def not_raises(exception):
  try:
    yield
  except exception:
    raise pytest.fail("DID RAISE {0}".format(str(exception)))

def test_unify_int_int():
        Type_.int().unify({}, Type_.int())

def test_unify_real_real():
        Type_.real().unify({}, Type_.real())

def test_unify_int_real():
        with pytest.raises(IncompatibleTypes):
                Type_.int().unify({}, Type_.real())

def test_unify_real_int():
        with pytest.raises(IncompatibleTypes):
                Type_.real().unify({}, Type_.int())

def test_unify_array_real():
        with pytest.raises(IncompatibleTypes):
                Type_.array(Type_.real()).unify({}, Type_.real())

def test_unify_array_vector():
        with pytest.raises(IncompatibleTypes):
                Type_.array(Type_.real()).unify({}, Type_.vector())

def test_unify_array_indexed():
        Type_.array(Type_.real()).unify({}, Type_.indexed())

def test_unify_array_indexed_real():
        Type_.array(Type_.real()).unify({}, Type_.indexed(Type_.real()))

def test_unify_array_indexed_int():
        with pytest.raises(IncompatibleTypes):
                Type_.array(Type_.real()).unify({}, Type_.indexed(Type_.int()))

def test_unify_vector_indexed_real():
        Type_.vector().unify({}, Type_.indexed(Type_.real()))

def test_unify_vector_indexed_int():
        with pytest.raises(IncompatibleTypes):
                Type_.vector().unify({}, Type_.indexed(Type_.int()))

def test_unify_matrix_indexed1():
        Type_.matrix().unify({}, Type_.indexed())

def test_unify_matrix_indexed_real1():
        with pytest.raises(IncompatibleTypes):
                Type_.matrix().unify({}, Type_.indexed(Type_.real()))

def test_unify_matrix_indexed2():
        Type_.matrix().unify({}, Type_.indexed(Type_.indexed()))

def test_unify_matrix_indexed_real2():
        Type_.matrix().unify({}, Type_.indexed(Type_.indexed(Type_.real())))

def test_unify_matrix_indexed3():
        with pytest.raises(IncompatibleTypes):
                Type_.matrix().unify({}, Type_.indexed(Type_.indexed(Type_.indexed())))

def test_unify_matrix_indexed_real3():
        with pytest.raises(IncompatibleTypes):
                Type_.matrix().unify({}, Type_.indexed(Type_.indexed(Type_.indexed(Type_.real()))))

def test_unify_variable_int():
        Type_.newVariable().unify({}, Type_.indexed(Type_.int()))

def test_unify_variable_indexed():
        Type_.newVariable().unify({}, Type_.indexed(Type_.int()))

def test_unify_named_variable_int():
        env = {}
        Type_.namedVariable(env, "x").unify(env, Type_.indexed(Type_.int()))

def test_unify_named_variable_indexed():
        env = {}
        Type_.namedVariable(env, "x").unify(env, Type_.indexed(Type_.int()))

def test_unify_named_variable_variable():
        env = {}
        Type_.namedVariable(env, "x").unify(env, Type_.newVariable())

def test_unify_named_variable_named_variable_same():
        env = {}
        Type_.namedVariable(env, "x").unify(env, Type_.namedVariable(env, "x"))

def test_unify_named_variable_named_variable_different():
        env = {}
        Type_.namedVariable(env, "x").unify(env, Type_.namedVariable(env, "y"))

def test_unify_named_variable_named_variable_different_no_conflict():
        env = {}
        Type_.namedVariable(env, "x").unify(env, Type_.int())
        Type_.namedVariable(env, "y").unify(env, Type_.real())

def test_unify_named_variable_named_variable_same_conflict():
        with pytest.raises(IncompatibleTypes):
                env = {}
                Type_.namedVariable(env, "x").unify(env, Type_.int())
                Type_.namedVariable(env, "x").unify(env, Type_.real())

def test_unify_named_variable_named_variable_different_conflict():
        with pytest.raises(IncompatibleTypes):
                env = {}
                Type_.namedVariable(env, "x").unify(env, Type_.int())
                Type_.namedVariable(env, "y").unify(env, Type_.real())
                Type_.namedVariable(env, "y").unify(env, Type_.namedVariable(env, "x"))

def test_unify_matrix_indexed_named_variable_real():
        env = {}
        Type_.matrix().unify(env, Type_.indexed(Type_.namedVariable(env, "x")))
        Type_.indexed(Type_.namedVariable(env, "y")).unify(env, Type_.namedVariable(env, "x"))
        Type_.real().unify(env, Type_.namedVariable(env, "y"))

def test_unify_matrix_indexed_named_variable_int():
        with pytest.raises(IncompatibleTypes):
                env = {}
                Type_.matrix().unify(env, Type_.indexed(Type_.namedVariable(env, "x")))
                Type_.indexed(Type_.namedVariable(env, "y")).unify(env, Type_.namedVariable(env, "x"))
                Type_.int().unify(env, Type_.namedVariable(env, "y"))
