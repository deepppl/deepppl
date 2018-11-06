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
from deepppl.translation.sdim import Dimension, \
        DnewVariable, DnamedVariable, DpathDimension
from deepppl.translation.stype import Type_, \
        Treal, Tint, Tindexed, Tvector, Trow_vector, Tmatrix, Tarray, \
        TnamedVariable, TnewVariable

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

### This section tests unification of dimensions
def test_unify_dim_variable_variable():
        DnewVariable().unify(DnewVariable())

def test_unify_dim_variable_named_variable():
        DnewVariable().unify(DnamedVariable({}, "x"))

def test_unify_dim_named_variable_named_variable_same_wrong_envs():
        DnamedVariable({}, "x").unify(DnamedVariable({}, "x"))

def test_unify_dim_named_variable_named_variable_same():
        denv={}
        DnamedVariable(denv, "x").unify(DnamedVariable(denv, "x"), denv=denv)

def test_unify_dim_named_variable_named_variable_different():
        denv={}
        DnamedVariable(denv, "x").unify(DnamedVariable(denv, "y"), denv=denv)

def test_unify_dim_variable_path():
        DnewVariable().unify(DpathDimension("mlp.size"))

def test_unify_dim_named_variable_path():
        denv = {}
        DnamedVariable(denv, "x").unify(DpathDimension("mlp.size"))

def test_unify_dim_path_path_same():
        DpathDimension("mlp.size").unify(DpathDimension("mlp.size"))

def test_unify_dim_path_path_different():
        with pytest.raises(IncompatibleDimensions):
                DpathDimension("mlp.size").unify(DpathDimension("mlp.other_size"))

def test_unify_dim_path_path_different_via_named():
        with pytest.raises(IncompatibleDimensions):
                denv = {}
                DnamedVariable(denv, "x").unify(DpathDimension("mlp.size"),denv=denv)
                DpathDimension("mlp.size_other").unify(DnamedVariable(denv, "y"),denv=denv)
                DnamedVariable(denv, "x").unify(DnamedVariable(denv, "y"),denv=denv)

### This section tests unification of types (with anonymous dimensions)
def test_unify_int_int():
        Tint().unify(Tint())

def test_unify_real_real():
        Treal().unify(Treal())

def test_unify_int_real():
        with pytest.raises(IncompatibleTypes):
                Tint().unify(Treal())

def test_unify_real_int():
        with pytest.raises(IncompatibleTypes):
                Treal().unify(Tint())

def test_unify_array_real():
        with pytest.raises(IncompatibleTypes):
                Tarray(Treal()).unify(Treal())

def test_unify_array_vector():
        with pytest.raises(IncompatibleTypes):
                Tarray(Treal()).unify(Tvector())

def test_unify_array_indexed():
        Tarray(Treal()).unify(Tindexed())

def test_unify_array_indexed_real():
        Tarray(Treal()).unify(Tindexed(Treal()))

def test_unify_array_indexed_int():
        with pytest.raises(IncompatibleTypes):
                Tarray(Treal()).unify(Tindexed(Tint()))

def test_unify_vector_indexed_real():
        Tvector().unify(Tindexed(Treal()))

def test_unify_vector_indexed_int():
        with pytest.raises(IncompatibleTypes):
                Tvector().unify(Tindexed(Tint()))

def test_unify_matrix_indexed1():
        Tmatrix().unify(Tindexed())

def test_unify_matrix_indexed_real1():
        with pytest.raises(IncompatibleTypes):
                Tmatrix().unify(Tindexed(Treal()))

def test_unify_matrix_indexed2():
        Tmatrix().unify(Tindexed(Tindexed()))

def test_unify_matrix_indexed_real2():
        Tmatrix().unify(Tindexed(Tindexed(Treal())))

def test_unify_matrix_indexed3():
        with pytest.raises(IncompatibleTypes):
                Tmatrix().unify(Tindexed(Tindexed(Tindexed())))

def test_unify_matrix_indexed_real3():
        with pytest.raises(IncompatibleTypes):
                Tmatrix().unify(Tindexed(Tindexed(Tindexed(Treal()))))

def test_unify_variable_int():
        TnewVariable().unify(Tindexed(Tint()))

def test_unify_variable_indexed():
        TnewVariable().unify(Tindexed(Tint()))

def test_unify_named_variable_int():
        tenv = {}
        TnamedVariable(tenv, "x").unify(Tindexed(Tint()), tenv=tenv)

def test_unify_named_variable_indexed():
        tenv = {}
        TnamedVariable(tenv, "x").unify(Tindexed(Tint()), tenv=tenv)

def test_unify_named_variable_variable():
        tenv = {}
        TnamedVariable(tenv, "x").unify(TnewVariable(), tenv=tenv)

def test_unify_named_variable_named_variable_same():
        tenv = {}
        TnamedVariable(tenv, "x").unify(TnamedVariable(tenv, "x"), tenv=tenv)

def test_unify_named_variable_named_variable_different():
        tenv = {}
        TnamedVariable(tenv, "x").unify(TnamedVariable(tenv, "y"), tenv=tenv)

def test_unify_named_variable_named_variable_different_no_conflict():
        tenv = {}
        TnamedVariable(tenv, "x").unify(Tint(), tenv=tenv)
        TnamedVariable(tenv, "y").unify(Treal(), tenv=tenv)

def test_unify_named_variable_named_variable_same_conflict():
        with pytest.raises(IncompatibleTypes):
                tenv = {}
                TnamedVariable(tenv, "x").unify(Tint(), tenv=tenv)
                TnamedVariable(tenv, "x").unify(Treal(), tenv=tenv)

def test_unify_named_variable_named_variable_different_conflict():
        with pytest.raises(IncompatibleTypes):
                tenv = {}
                TnamedVariable(tenv, "x").unify(Tint(), tenv=tenv)
                TnamedVariable(tenv, "y").unify(Treal(), tenv=tenv)
                TnamedVariable(tenv, "y").unify(TnamedVariable(tenv, "x"), tenv=tenv)

def test_unify_matrix_indexed_named_variable_real():
        tenv = {}
        Tmatrix().unify(Tindexed(TnamedVariable(tenv, "x")), tenv=tenv)
        Tindexed(TnamedVariable(tenv, "y")).unify(TnamedVariable(tenv, "x"), tenv=tenv)
        Treal().unify(TnamedVariable(tenv, "y"), tenv=tenv)

def test_unify_matrix_indexed_named_variable_int():
        with pytest.raises(IncompatibleTypes):
                tenv = {}
                Tmatrix().unify(Tindexed(TnamedVariable(tenv, "x")), tenv=tenv)
                Tindexed(TnamedVariable(tenv, "y")).unify(TnamedVariable(tenv, "x"), tenv=tenv)
                Tint().unify(TnamedVariable(tenv, "y"), tenv=tenv)
