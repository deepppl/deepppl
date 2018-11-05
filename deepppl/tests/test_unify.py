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
from deepppl.translation.stype import Type_, \
        Sreal, Sint, Sindexed, Svector, Srow_vector, Smatrix, Sarray, \
        SnamedVariable, SnewVariable

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
        Sint().unify(Sint())

def test_unify_real_real():
        Sreal().unify(Sreal())

def test_unify_int_real():
        with pytest.raises(IncompatibleTypes):
                Sint().unify(Sreal())

def test_unify_real_int():
        with pytest.raises(IncompatibleTypes):
                Sreal().unify(Sint())

def test_unify_array_real():
        with pytest.raises(IncompatibleTypes):
                Sarray(Sreal()).unify(Sreal())

def test_unify_array_vector():
        with pytest.raises(IncompatibleTypes):
                Sarray(Sreal()).unify(Svector())

def test_unify_array_indexed():
        Sarray(Sreal()).unify(Sindexed())

def test_unify_array_indexed_real():
        Sarray(Sreal()).unify(Sindexed(Sreal()))

def test_unify_array_indexed_int():
        with pytest.raises(IncompatibleTypes):
                Sarray(Sreal()).unify(Sindexed(Sint()))

def test_unify_vector_indexed_real():
        Svector().unify(Sindexed(Sreal()))

def test_unify_vector_indexed_int():
        with pytest.raises(IncompatibleTypes):
                Svector().unify(Sindexed(Sint()))

def test_unify_matrix_indexed1():
        Smatrix().unify(Sindexed())

def test_unify_matrix_indexed_real1():
        with pytest.raises(IncompatibleTypes):
                Smatrix().unify(Sindexed(Sreal()))

def test_unify_matrix_indexed2():
        Smatrix().unify(Sindexed(Sindexed()))

def test_unify_matrix_indexed_real2():
        Smatrix().unify(Sindexed(Sindexed(Sreal())))

def test_unify_matrix_indexed3():
        with pytest.raises(IncompatibleTypes):
                Smatrix().unify(Sindexed(Sindexed(Sindexed())))

def test_unify_matrix_indexed_real3():
        with pytest.raises(IncompatibleTypes):
                Smatrix().unify(Sindexed(Sindexed(Sindexed(Sreal()))))

def test_unify_variable_int():
        SnewVariable().unify(Sindexed(Sint()))

def test_unify_variable_indexed():
        SnewVariable().unify(Sindexed(Sint()))

def test_unify_named_variable_int():
        tenv = {}
        SnamedVariable(tenv, "x").unify(Sindexed(Sint()), tenv=tenv)

def test_unify_named_variable_indexed():
        tenv = {}
        SnamedVariable(tenv, "x").unify(Sindexed(Sint()), tenv=tenv)

def test_unify_named_variable_variable():
        tenv = {}
        SnamedVariable(tenv, "x").unify(SnewVariable(), tenv=tenv)

def test_unify_named_variable_named_variable_same():
        tenv = {}
        SnamedVariable(tenv, "x").unify(SnamedVariable(tenv, "x"), tenv=tenv)

def test_unify_named_variable_named_variable_different():
        tenv = {}
        SnamedVariable(tenv, "x").unify(SnamedVariable(tenv, "y"), tenv=tenv)

def test_unify_named_variable_named_variable_different_no_conflict():
        tenv = {}
        SnamedVariable(tenv, "x").unify(Sint(), tenv=tenv)
        SnamedVariable(tenv, "y").unify(Sreal(), tenv=tenv)

def test_unify_named_variable_named_variable_same_conflict():
        with pytest.raises(IncompatibleTypes):
                tenv = {}
                SnamedVariable(tenv, "x").unify(Sint(), tenv=tenv)
                SnamedVariable(tenv, "x").unify(Sreal(), tenv=tenv)

def test_unify_named_variable_named_variable_different_conflict():
        with pytest.raises(IncompatibleTypes):
                tenv = {}
                SnamedVariable(tenv, "x").unify(Sint(), tenv=tenv)
                SnamedVariable(tenv, "y").unify(Sreal(), tenv=tenv)
                SnamedVariable(tenv, "y").unify(SnamedVariable(tenv, "x"), tenv=tenv)

def test_unify_matrix_indexed_named_variable_real():
        tenv = {}
        Smatrix().unify(Sindexed(SnamedVariable(tenv, "x")), tenv=tenv)
        Sindexed(SnamedVariable(tenv, "y")).unify(SnamedVariable(tenv, "x"), tenv=tenv)
        Sreal().unify(SnamedVariable(tenv, "y"), tenv=tenv)

def test_unify_matrix_indexed_named_variable_int():
        with pytest.raises(IncompatibleTypes):
                tenv = {}
                Smatrix().unify(Sindexed(SnamedVariable(tenv, "x")), tenv=tenv)
                Sindexed(SnamedVariable(tenv, "y")).unify(SnamedVariable(tenv, "x"), tenv=tenv)
                Sint().unify(SnamedVariable(tenv, "y"), tenv=tenv)
