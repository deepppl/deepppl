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

from collections import defaultdict, OrderedDict
#from contextlib import contextmanager
import ast
#import torch
#import astpretty
import astor
#import sys
from .ir import IR, Program, ProgramBlocks, Data, VariableDecl, Subscript, \
                NetVariable, Program, ForStmt, ConditionalStmt, \
                AssignStmt, Subscript, BlockStmt,\
                CallStmt, List, SamplingStmt, SamplingDeclaration, SamplingObserved,\
                SamplingParameters, Variable, Constant, BinaryOperator, \
                Minus, UnaryOperator, UMinus, AnonymousShapeProperty,\
                VariableProperty, NetVariableProperty, Prior

from .ir import Type_ as IrType

from .sdim import Dimension, \
        Dnew, Dnamed, Dpath, Dconstant
from .stype import Type_, \
        Treal, Tint, Tindexed, Tvector, Trow_vector, Tmatrix, Tarray, \
        Tnamed, Tnew

from .exceptions import *

from_test = lambda: hasattr(sys, "_called_from_test")
class IRVisitor(object):
    def defaultVisit(self, node):
        raise NotImplementedError

    def __getattr__(self, attr):
        if attr.startswith('visit'):
            return self.defaultVisit
        return self.__getattribute__(attr)

    def _visitAll(self, iterable):
        filter = lambda x: (x is not None) or None
        return [filter(x) and x.accept(self) for x in iterable ]

    def _visitChildren(self, node):
        return self._visitAll(node.children)

class DimensionInferenceVisitor(IRVisitor):
    def __init__(self, outer:'TypeInferenceVisitor'):
        super(DimensionInferenceVisitor, self).__init__()
        self.outer = outer

    def visitVariable(self, var:Variable):
        d = Dnamed(self.outer.denv, var.id)
        var.expr_dim = d
        return d


    def visitVariableProperty(self, prop:VariableProperty) -> Dimension:
        if prop.prop != 'shape':
            raise UnsupportedProperty(prop)
        t = Tnamed(self.outer.tenv, prop.var.id)
        ## TODO: this does not handle anonymous type variables properly
        ## But this is probably not a problem in practice, since types need to be declared with a type

        ## TODO: this does not allow vector/matrix shapes.  This is by design, however it might be the wrong design.
        ## If they are to be handled, we need to decide what the shape of int[3] x[4] should be.
        if not t.isArray():
            raise IncompatibleTypes(t, Tindexed(Tnew()))
        else:
            d = t.dimensions()

        prop.expr_dim = d
        return d


class TypeInferenceVisitor(IRVisitor):
    """ Running 
    """

    @classmethod
    def run(cls, ir:IR):
        denv = {}
        tenv = {}
        visitor = cls(tenv=tenv, denv=denv)
        ir.accept(visitor)
        return visitor

    def __init__(self, *, denv={}, tenv={}):
        super(TypeInferenceVisitor, self).__init__()
        self._ctx = {}
        self._anons = {}
        self._nets = {}
        self.denv = denv
        self.tenv = tenv

    known_functions = set([
        "randn", "exp", "log", "zeros", "ones", "softplus"
    ])

    def inferDims(self, ir:IR):
        return ir.accept(DimensionInferenceVisitor(self))

    def visitCallStmt(self, stmt:CallStmt):
        if stmt.id in set(["exp"]):
            arg0_type = stmt.args.children[0].accept(self)
            res = arg0_type.asRealArray()
        elif stmt.id in set(["zeros", "ones"]):
            args = stmt.args.children
            if len(args) == 0:
                assert False, "TODO: add code for adding in an anonymous dimension here"

            else:
            ## TODO: handle multiple explicit dimensions here
                res = Treal()
                for arg in stmt.args.children:
                    dim = self.inferDims(arg)
                    res = Tarray(component = res, dimension=dim)
        else:
            res = Tnew()

            if stmt.id in self.known_functions:
                if len(stmt.args.children) != 0:
                    for a in stmt.args.children:
                        self.Tunify(res, a.accept(self))

        stmt.expr_type = res
        return res

    def visitSamplingStmt(self, stmt:SamplingStmt):
        target_type = stmt.target.accept(self)

        if stmt.id == 'Normal':
            assert len(stmt.args) >= 2, f"Normal distribution underspecified; only {len(stmt.args)} arguments given"
            assert len(stmt.args) <= 3, f"Normal distribution overspecified; {len(stmt.args)} arguments given"
            # TODO: this accepts int arrays.  Is that actually valid?
            t0 = stmt.args[0].accept(self).asRealArray()
            t1 = stmt.args[1].accept(self).asRealArray()
            self.Tunify(t0, t1)
            if len(stmt.args) == 3:
                t2 = stmt.args[2].accept(self)
                self.Tunify(t2, Tint())
                # TODO: This is wrong.  How do we find the right shape?
                # Do we need a separate visitor for dimensions?
                t0 = Tarray(dimension=Dnew(), component=t0)
            self.Tunify(target_type, t0)
        elif stmt.id == 'Bernoulli':
            assert len(stmt.args) == 1, f"Bernoulli distribution expected to have 1 argument; {len(stmt.args)} arguments given"
            t0 = stmt.args[0].accept(self).realArrayToIntArray()
            self.Tunify(target_type, t0)
        else:
            for a in stmt.args:
                self.Tunify(target_type, a.accept(self))
        stmt.expr_type = target_type
        return target_type

    visitSamplingParameters = visitSamplingStmt
    visitSamplingObserved = visitSamplingStmt

    def Tunify(self, t1:Type_, t2:Type_):
        t1.unify(t2, denv=self.denv, tenv=self.tenv)

    def visitProgram(self, program:Program):
        for b in program.children:
            b.accept(self)
        
    def visitProgramBlocks(self, blocks:ProgramBlocks):
        for b in blocks.children:
            b.accept(self)

    visitData = visitProgramBlocks
    visitParameters = visitProgramBlocks
    visitGuideParameters = visitProgramBlocks
    visitSamplingBlocks = visitProgramBlocks
    visitModel = visitSamplingBlocks
    visitGuide = visitSamplingBlocks
    visitPrior = visitSamplingBlocks
    
    def visitNetworksBlock(self, blocks):
        pass

    visitGuide = visitProgramBlocks

    def toSType(self, t:IrType):
        if t.type_ == 'int':
            return Tint()
        elif t.type_ == 'real':
            return Treal()
        else:
            assert False, f"Unknown type: {self.type_}"

    def toSDim(self, d):
        if isinstance(d, Constant):
            return Dconstant(d.value)
        elif isinstance(d, Variable):
            # add support for named dimension variables here
            return Dnamed(self.denv, d.id)
        else:
            # TODO: add support for anonymous parameters
            assert False, f"Unknown dimension type: {self.type_}"

    def visitVariableDecl(self, decl:VariableDecl):
        var_type:Type_ = self.toSType(decl.type_)
        if decl.dim:
            if isinstance(decl.dim, Variable):
                dim_type = self.toSDim(decl.dim)
                var_type = Tarray(component=var_type, dimension=dim_type)
            elif decl.dim.children:
                for d in decl.dim.children:
                    dim_type = self.toSDim(d)
                    var_type = Tarray(component=var_type, dimension=dim_type)
        var = Tnamed(self.tenv, decl.id)
        self.Tunify(var, var_type)
        if decl.init:
            init_type = decl.init.accept(self)
            self.Tunify(var, init_type)
        decl.expr_type = var

    def visitVariable(self, var:Variable):
        t = Tnamed(self.tenv, var.id)
        var.expr_type = t
        return t


    def visitVariableProperty(self, prop:VariableProperty) -> Dimension:
        assert False, "coding error"
        if prop.prop != 'shape':
            raise UnsupportedProperty(prop)
        t = Tnamed(self.tenv, prop.var.id)

        prop.expr_type = t
        return t


    def visitAssignStmt(self, stmt:AssignStmt):
        target = stmt.target.accept(self)
        value = stmt.value.accept(self)

        self.Tunify(target, value)

    def visitSubscript(self, expr:Subscript):
        id_type = expr.id.accept(self)
        used = len(expr.index.exprs) if expr.index.is_tuple() else 1
        base = Tnew()
        c = base
        for _ in range(used):
            c = Tindexed(component=c)
        self.Tunify(c, id_type)
        expr.expr_type = base
        return base

""" 
    def visitNetVariableProperty(self, netprop):
        net = netprop.var
        prop = netprop.prop
        if prop != 'shape':
            raise UnsupportedProperty(prop)
        name = '.'.join([net.name,] + net.ids)
        if name in self._nets:
            answer = self._nets[name]
        else:
            answer = ShapeLinkedList.bounded(netprop, name)
            self._nets[name] = answer
        return answer

 """