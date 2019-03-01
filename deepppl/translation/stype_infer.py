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

from typing import Union, Sequence
from collections import defaultdict, OrderedDict
#from contextlib import contextmanager
import ast
#import torch
#import astpretty
import astor
#import sys
from .ir import IR, Program, ProgramBlocks, Data, VariableDecl, Subscript, \
                NetVariable, NetDeclaration, Program, ForStmt, ConditionalStmt, \
                AssignStmt, Subscript, BlockStmt,\
                CallStmt, List, SamplingStmt, SamplingDeclaration, SamplingObserved,\
                SamplingParameters, Variable, Constant, BinaryOperator, \
                Minus, UnaryOperator, UMinus, AnonymousShapeProperty,\
                VariableProperty, NetVariableProperty, Prior

from .ir import Type_ as IrType

from .sdim import KnownDimension, Dimension, \
        Dnew, Dnamed, Dshape, Druntime, Dconstant, Groups
from .stype import Type_, \
        Treal, Tint, Tindexed, Tvector, Trow_vector, Tmatrix, Tarray, \
        Tnamed, Tnew, Tnetwork, Ttensor

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
        d = Druntime(var.id)
        var.expr_dim = d
        return d

    visitNetVariable = visitVariable

    def visitAnonymousShapeProperty(self, prop:AnonymousShapeProperty):
        d = Dnamed(self.outer.denv, prop.var.id)

        prop.expr_dim = d
        return d

    def visitVariableProperty(self, prop:VariableProperty) -> Dimension:
        if prop.prop != 'shape':
            raise UnsupportedProperty(prop)
        t = Tnamed(self.outer.tenv, prop.var.id)
        ## TODO: this does not handle anonymous type variables properly
        ## But this is probably not a problem in practice, since types need to be declared with a type

        ## TODO: this does not allow vector/matrix shapes.  This is by design, however it might be the wrong design.
        ## If they are to be handled, we need to decide what the shape of int[3] x[4] should be.
        if not t.isArray() and not t.isPrimitive():
            raise IncompatibleTypes(t, Tindexed(Tnew()))
        else:
            d = t.dimensions()

        prop.expr_dim = d
        return d

    def visitNetVariableProperty(self, prop:NetVariableProperty) -> Union[Sequence[Dimension], Dimension]:
        if prop.prop != 'shape':
            raise UnsupportedProperty(prop)
        t = self.outer._nets[prop.var.name][prop.var.ids]
        ## TODO: this does not handle anonymous type variables properly
        ## But this is probably not a problem in practice, since types need to be declared with a type

        ## TODO: this does not allow vector/matrix shapes.  This is by design, however it might be the wrong design.
        ## If they are to be handled, we need to decide what the shape of int[3] x[4] should be.
        # TODO: this does not work for network types
        # which happens, e.g. with the shape1 example
        if not t.isArray():
            raise IncompatibleTypes(t, Tindexed(Tnew()))
        else:
            d = t.dimensions()

        prop.expr_dim = d
        return d
class NetworkVariableType(object):
    def __init__(self, net_cls:str):
        self.net_cls = net_cls
        # self.input = input
        # self.output = output
        self._paths = {}            

    def __getitem__(self, paramPath:Union[str,Sequence[str]])->Type_:
        k = tuple(paramPath)
        return self._paths[k]

    def __setitem__(self, paramPath:Union[str,Sequence[str]], v:Type_)->Type_:
        k = tuple(paramPath)
        self._paths[k] = v

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

    def __init__(self, *, denv={}, tenv={}, equalities=Groups()):
        super(TypeInferenceVisitor, self).__init__()
        self._ctx = {}
        self._anons = {}
        self._nets = {}
        self.equalities = equalities
        self.denv = denv
        self.tenv = tenv

    def inferDims(self, ir:IR):
        return ir.accept(DimensionInferenceVisitor(self))

    def visitCallStmt(self, stmt:CallStmt):
        if stmt.id in set(["exp", "softplus", "log"]):
            arg0_type = stmt.args.children[0].accept(self)
            res = arg0_type.asRealArray()
        elif stmt.id in set(["zeros", "ones", "randn", "rand"]):
            args = stmt.args.children
            if len(args) == 0:
                stmt.args.children = [AnonymousShapeProperty()]
            ## TODO: handle multiple explicit dimensions here
            res = Treal()
            for arg in stmt.args.children:
                dim = self.inferDims(arg)
                res = Tarray(component = res, dimension=dim)
        elif stmt.id in self._nets:
            # right now we do not impose any constraints on the input
            # but we still iterate through them (so that constants get generalized, for example)
            for a in stmt.args.children:
                a.accept(self)
            res = Tnetwork(stmt)
        else:
            for a in stmt.args.children:
                a.accept(self)
            # TODO: can we do better?
            res = Tnew()

            # if stmt.id in self.known_functions:
            #     if len(stmt.args.children) != 0:
            #         for a in stmt.args.children:
            #             self.Tunify(res, a.accept(self))

        stmt.expr_type = res
        str(res)
        return res

    def visitSamplingStmt(self, stmt:SamplingStmt):
        target_type = stmt.target.accept(self)

        if stmt.id == 'normal':
            assert len(stmt.args) >= 2, f"Normal distribution underspecified; only {len(stmt.args)} arguments given"
            assert len(stmt.args) <= 3, f"Normal distribution overspecified; {len(stmt.args)} arguments given"
            # TODO: this accepts int arrays.  Is that actually valid?

            # TODO: if given an int array, then add an IR node to change
            # the int array into a real array to make pyro happy
            t0 = stmt.args[0].accept(self).asRealArray()
            t1 = stmt.args[1].accept(self).asRealArray()
            self.Tunify(t0, t1)
            if len(stmt.args) == 3:
                t2 = stmt.args[2].accept(self)
                self.Tunify(t2, Tint())
                # TODO: This is wrong.  How do we find the right shape?
                # Do we need a separate visitor for dimensions?
                t0 = Tarray(dimension=Dnew(), component=t0)
            res = Ttensor(t0)
            self.Tunify(target_type, res)
        elif stmt.id == 'bernoulli':
            assert len(stmt.args) == 1, f"Bernoulli distribution expected to have 1 argument; {len(stmt.args)} arguments given"
            arg0 = stmt.args[0].accept(self)
            t0 = arg0.realArrayToIntArray()
            res = Ttensor(t0)
            self.Tunify(target_type, res)
        elif stmt.id == 'beta':
            assert len(stmt.args) == 2, f"Beta distribution expected to have 2 argument; {len(stmt.args)} arguments given"
            alpha = stmt.args[0].accept(self).asRealArray()
            beta = stmt.args[1].accept(self).asRealArray()
            self.Tunify(alpha, beta)
            res = Ttensor(alpha)
            self.Tunify(target_type, res)
        # Fake distributions created by the translation
        elif stmt.id == 'ImproperUniform':
            assert len(stmt.args) < 2, f"ImproperUniform distribution expected to have at most 1 argument; {len(stmt.args)} arguments given"
            if len(stmt.args) == 0:
                stmt.args = [AnonymousShapeProperty()]
            sh = stmt.args[0]
            assert isinstance(sh, VariableProperty), "unknown shape type given to ImproperUniform"
            sh_var = sh.var
            sh_type = sh_var.accept(self)
            sh_type_real = sh_type.asRealArray()
            self.Tunify(target_type, sh_type_real)
        elif stmt.id == 'LowerConstrainedImproperUniform':
            assert len(stmt.args) <= 2, f"LowerConstrainedImproperUniform distribution expected to have at most 2 argument; {len(stmt.args)} arguments given"
            if len(stmt.args) == 1:
                stmt.args.append(AnonymousShapeProperty())

            lower = stmt.args[0].accept(self).asRealArray()
            lower_vec = Ttensor(lower)
            sh = stmt.args[1]
            assert isinstance(sh, VariableProperty), "unknown shape type given to LowerConstrainedImproperUniform"
            sh_var = sh.var
            sh_type = sh_var.accept(self)
            sh_type_real = sh_type.asRealArray()

            self.Tunify(lower_vec, sh_type_real)
            self.Tunify(target_type, sh_type_real)
        elif stmt.id == 'UpperConstrainedImproperUniform':
            assert len(stmt.args) <= 2, f"UpperConstrainedImproperUniform distribution expected to have at most 2 argument; {len(stmt.args)} arguments given"
            if len(stmt.args) == 1:
                stmt.args.append(AnonymousShapeProperty())

            upper = stmt.args[0].accept(self).asRealArray()
            upper_vec = Ttensor(upper)
            sh = stmt.args[1]
            assert isinstance(sh, VariableProperty), "unknown shape type given to UpperConstrainedImproperUniform"
            sh_var = sh.var
            sh_type = sh_var.accept(self)
            sh_type_real = sh_type.asRealArray()

            self.Tunify(upper_vec, sh_type_real)
            self.Tunify(target_type, sh_type_real)

        elif stmt.id == "Uniform":
            assert len(stmt.args) <= 3, f"Uniform distribution expected to have 2 argument; {len(stmt.args)} arguments given"
            if len(stmt.args) == 2:
                stmt.args.append(AnonymousShapeProperty())          
            lower = stmt.args[0].accept(self).asRealArray()
            upper = stmt.args[1].accept(self).asRealArray()
            sh = stmt.args[2]
            assert isinstance(sh, VariableProperty), "unknown shape type given to Uniform"
            sh_var = sh.var
            sh_type = sh_var.accept(self)
            sh_type_real = sh_type.asRealArray()

            self.Tunify(lower, upper)
            lower_vec = Ttensor(lower)
            self.Tunify(lower_vec, sh_type_real)
            self.Tunify(target_type, sh_type_real)
        else:
            assert False, f"The {stmt.id} distribution is not yet supported."
            # for a in stmt.args:
            #     self.Tunify(target_type, a.accept(self))
        stmt.expr_type = target_type
        return target_type

    visitSamplingParameters = visitSamplingStmt
    visitSamplingObserved = visitSamplingStmt
    visitSamplingDeclaration = visitSamplingStmt

    def visitNetDeclaration(self, decl:NetDeclaration):
        # TODO: Question: do all instantiations of a net (decl.name(x))
        # Share the same input/output type?  If so, then
        # We should stash the types here
        types = NetworkVariableType(decl.net_cls)
        for p in decl.params:
            types[p] = Tnetwork(".".join([decl.name] + p))
        self._nets[decl.name] = types

    def Tunify(self, t1:Type_, t2:Type_):
        t1.unify(t2, equalities=self.equalities, tenv=self.tenv)

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
    
    visitNetworksBlock = visitProgramBlocks

    visitGuide = visitProgramBlocks

    def visitBlockStmt(self, blocks:BlockStmt):
        for b in blocks.children:
            b.accept(self)

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
            return Druntime(d.id)
        elif isinstance(d, AnonymousShapeProperty):
            # TODO: really, this should probably be an actual anonymous dimension
            return Dnamed(self.denv, d.var.id)
        elif isinstance(d, VariableProperty):
            if d.prop != 'shape':
                raise UnsupportedProperty(d.prop)
            return Dshape(d.var.id)
        else:
            assert False, f"Unknown dimension type: {self.type_}"

    def visitVariableDecl(self, decl:VariableDecl):
        var_type:Type_ = self.toSType(decl.type_)
        if decl.dim:
            if isinstance(decl.dim, Variable) or isinstance(decl.dim, AnonymousShapeProperty) or isinstance(decl.dim, Constant) or isinstance(decl.dim, VariableProperty):
                dim_type = self.toSDim(decl.dim)
                var_type = Tarray(component=var_type, dimension=dim_type)
            elif decl.dim.children:
                for d in reversed(decl.dim.children):
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

    def visitNetVariable(self, var:Variable):
        netTypes = self._nets[var.name]
        return netTypes[var.ids]

    def visitVariableProperty(self, prop:VariableProperty) -> Dimension:
        assert False, "coding error"
        if prop.prop != 'shape':
            raise UnsupportedProperty(prop)
        t = Tnamed(self.tenv, prop.var.id)

        prop.expr_type = t
        return t

    visitNetVariableProperty = visitVariableProperty

    def visitForStmt(self, stmt:ForStmt):
        from_type = stmt.from_.accept(self)
        to_type = stmt.to_.accept(self)
        stmt.body.accept(self)

        self.Tunify(from_type, Tint())
        self.Tunify(to_type, Tint())
    
    def visitConditionalStmt(self, stmt:ConditionalStmt):

# We currently don't look at the test condition
#        test_type = stmt.test.accept(self)
        stmt.true.accept(self)
        stmt.false.accept(self)

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

    def visitConstant(self, c:Constant):
        t = Tnew("const")
        c.expr_type = t
        return t

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