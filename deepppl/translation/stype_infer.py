
# /* Copyright 2018 
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
                Plus, Minus, Mult, DotMult, Div, DotDiv, UnaryOperator, UPlus, UMinus, AnonymousShapeProperty,\
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

    def visitConstant(self, const:Constant):
        v = const.value
        if isinstance(v, (int, float)):
            return []
        assert False, f"We do not yet handle constant non-flat shapes {const.value}"

    def visitAnonymousShapeProperty(self, prop:AnonymousShapeProperty):
        d = Dnamed(self.outer.denv, prop.var.id)

        prop.expr_dim = d
        return d

    def visitVariableProperty(self, prop:VariableProperty):
        if prop.prop != 'shape':
            raise UnsupportedProperty(prop)
        t = Tnamed(self.outer.tenv, prop.var.id)
        ## TODO: this does not handle anonymous type variables properly
        ## But this is probably not a problem in practice, since types need to be declared with a type

        ## TODO: this does not allow vector/matrix shapes.  This is by design, however it might be the wrong design.
        ## If they are to be handled, we need to decide what the shape of int[3] x[4] should be.
#            raise IncompatibleTypes(t, Tindexed(Tnew()))
#        else:
        d = t.dimensions()
        prop.expr_dim = t.all_dimensions()

        if t.isNonArrayIndexed():
            return (t, d)
        else:
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

    def __init__(self, *, denv={}, tenv={}, equalities=None):
        super(TypeInferenceVisitor, self).__init__()
        self._ctx = {}
        self._anons = {}
        self._nets = {}
        if equalities is None:
            equalities = Groups()
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
                if isinstance(dim, tuple):
                    res = dim[0]
                    dim = dim[1]
                if isinstance(dim, list):
                    dim = dim.copy()
                    dim.reverse()
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

    def dimToRealArray(self, sh):
        if isinstance(sh, VariableProperty):
            sh_var = sh.var
            sh_type = sh_var.accept(self)
            return sh_type.asRealArray()
        if isinstance(sh, Variable) or isinstance(sh, Constant) or isinstance(sh, VariableProperty):
            dim_type = self.toSDim(sh)
            var_type = Treal()
            return Tindexed(component=var_type, dimension=dim_type)
        elif sh.children:
            var_type = Treal()
            for d in reversed(sh.children):
                dim_type = self.toSDim(d)
                var_type = Tindexed(component=var_type, dimension=dim_type)
            return var_type

    def visitSamplingStmt(self, stmt:SamplingStmt):
        target_type = stmt.target.accept(self)

        if stmt.id == 'normal':
            assert len(stmt.args) >= 2, f"normal distribution underspecified; only {len(stmt.args)} arguments given"
            assert len(stmt.args) <= 2, f"normal distribution overspecified; {len(stmt.args)} arguments given"
            # TODO: this accepts int arrays.  Is that actually valid?

            # TODO: if given an int array, then add an IR node to change
            # the int array into a real array to make pyro happy
            t0 = stmt.args[0].accept(self).asRealArray()
            t1 = stmt.args[1].accept(self).asRealArray()
            # self.Tunify(t0, t1)
            if stmt.shape is not None:
                t2 = stmt.shape.accept(self)
                self.Tunify(t2, Tint())
                # TODO: This is wrong.  How do we find the right shape?
                # Do we need a separate visitor for dimensions?
                t0 = Tarray(dimension=Dnew(), component=t0)
            res = Ttensor(t0)
            self.Tunify(target_type, res)
        elif stmt.id == 'uniform':
            assert len(stmt.args) >= 2, f"uniform distribution underspecified; only {len(stmt.args)} arguments given"
            assert len(stmt.args) <= 2, f"uniform distribution overspecified; {len(stmt.args)} arguments given"
            # TODO: this accepts int arrays.  Is that actually valid?

            # TODO: if given an int array, then add an IR node to change
            # the int array into a real array to make pyro happy
            alpha = stmt.args[0].accept(self)
            beta = stmt.args[1].accept(self)
            # They both need to be reals
            self.Tunify(alpha, Ttensor(Treal()))
            self.Tunify(beta, Ttensor(Treal()))

            res = Ttensor(alpha)
            self.Tunify(target_type, res)
        elif stmt.id == 'bernoulli':
            assert len(stmt.args) == 1, f"bernoulli distribution expected to have 1 argument; {len(stmt.args)} arguments given"
            arg0 = stmt.args[0].accept(self)
            t0 = arg0.realArrayToIntArray()
            res = Ttensor(t0)
            self.Tunify(target_type, res)
        elif stmt.id == 'beta':
            assert len(stmt.args) == 2, f"beta distribution expected to have 2 argument; {len(stmt.args)} arguments given"
            alpha = stmt.args[0].accept(self).asRealArray()
            beta = stmt.args[1].accept(self).asRealArray()
            self.Tunify(alpha, beta)
            res = Ttensor(alpha)
            self.Tunify(target_type, res)
        elif stmt.id == 'categorical_logits':
            assert len(stmt.args) == 1, f"categorical_logits distribution expected to have 1 argument; {len(stmt.args)} arguments given"
            # Make sure that the result is over an integer
            self.Tunify(target_type, Ttensor(Tint()))

            # the result is one dimension less than the input (and not over a real)
            arg0 = stmt.args[0].accept(self)
            inp = Ttensor(component=target_type.asRealArray())
            self.Tunify(inp, arg0)


        # Fake distributions created by the translation
        elif stmt.id == 'ImproperUniform':
            assert len(stmt.args) == 0, f"ImproperUniform distribution expected to have at most 0 argument; {len(stmt.args)} arguments given"
            if stmt.shape is None:
                stmt.shape = AnonymousShapeProperty()
            sh = stmt.shape
            sh_type_real = self.dimToRealArray(sh)

            self.Tunify(target_type, sh_type_real)

        elif stmt.id == 'LowerConstrainedImproperUniform':
            assert len(stmt.args) == 1, f"LowerConstrainedImproperUniform distribution expected to one non shape argument; {len(stmt.args)} arguments given"
            if stmt.shape is None:
                stmt.shape = AnonymousShapeProperty()
            sh = stmt.shape
            sh_type_real = self.dimToRealArray(sh)

            lower = stmt.args[0].accept(self).asRealArray()
            lower_vec = Ttensor(lower)

            self.Tunify(lower_vec, sh_type_real)
            self.Tunify(target_type, sh_type_real)
        elif stmt.id == 'UpperConstrainedImproperUniform':
            assert len(stmt.args) == 1, f"UpperConstrainedImproperUniform distribution expected to have one non shape argument; {len(stmt.args)} arguments given"
            if stmt.shape is None:
                stmt.shape = AnonymousShapeProperty()
            sh = stmt.shape
            sh_type_real = self.dimToRealArray(sh)

            upper = stmt.args[0].accept(self).asRealArray()
            upper_vec = Ttensor(upper)

            self.Tunify(upper_vec, sh_type_real)
            self.Tunify(target_type, sh_type_real)

        elif stmt.id == "Uniform":
            assert len(stmt.args) == 2, f"Uniform distribution expected to have 2 non-shape argument; {len(stmt.args)} arguments given"
            if stmt.shape is None:
                stmt.shape = AnonymousShapeProperty()
            sh = stmt.shape
            sh_type_real = self.dimToRealArray(sh)

            lower = stmt.args[0].accept(self).asRealArray()
            upper = stmt.args[1].accept(self).asRealArray()

            self.Tunify(lower, upper)
            lower_vec = Ttensor(lower)
            self.Tunify(lower_vec, sh_type_real)
            self.Tunify(target_type, sh_type_real)
        elif stmt.id == 'Exponential':
            assert len(stmt.args) == 1, f"Exponential distribution expected to have 2 argument; {len(stmt.args)} arguments given"
            rate = stmt.args[0].accept(self).asRealArray()
            self.Tunify(rate, Ttensor(Treal()))

            res = Ttensor(rate)
            self.Tunify(target_type, res)

        else:
            print(f"WARNING: unknown distribution {stmt.id} is not yet supported")
            for arg in stmt.args:
                arg.accept(self)

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
    visitTransformedData = visitProgramBlocks
    visitParameters = visitProgramBlocks
    visitTransformedParameters = visitProgramBlocks
    visitGuideParameters = visitProgramBlocks
    visitSamplingBlocks = visitProgramBlocks
    visitGeneratedQuantities = visitProgramBlocks
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
        elif t.type_ == 'vector':
            dim1 = self.toSDim(t.dim)
            return Tvector(dimension=dim1)
        elif t.type_ == 'matrix':
            dim1 = self.toSDim(t.dim[0])
            dim2 = self.toSDim(t.dim[1])
            return Tmatrix(dimension1=dim1, dimension2=dim2)
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
        dims = decl.dim
        intrinsicDims = len(var_type.intrinsicDimensions())
        if dims:
            if intrinsicDims == 0 and isinstance(dims, AnonymousShapeProperty):
                var_type = Ttensor(var_type)
            elif intrinsicDims == 0 and (isinstance(dims, Variable) or isinstance(dims, Constant) or isinstance(dims, VariableProperty)):
                dim_type = self.toSDim(dims)
                var_type = Tarray(component=var_type, dimension=dim_type)
            elif dims.children:
                for d in reversed(dims.children):
                    if intrinsicDims > 0:
                        intrinsicDims = intrinsicDims - 1
                        continue
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
        if stmt.false is not None:
            stmt.false.accept(self)

    def visitAssignStmt(self, stmt:AssignStmt):
        target = stmt.target.accept(self)
        value = stmt.value.accept(self)

        self.Tunify(target, value)

    def visitBinaryOperator(self, op:BinaryOperator):
        if isinstance(op.op, (DotMult, DotDiv)):
            left = op.left.accept(self)
            right = op.right.accept(self)
            self.Tunify(left, right)
            op.expr_type = left
            return left
        if isinstance(op.op, (Plus, Minus)):
            left = op.left.accept(self)
            right = op.right.accept(self)
            if left.isPrimitive():
                op.expr_type = right
                return right
            if right.isPrimitive():
                op.expr_type = left
                return left
            if left.isArray() and right.isArray():
                tl = Ttensor(left)                
                tr = Ttensor(right)
                self.Tunify(tl, tr)
                rett = tl.description().component
                op.expr_type = rett
                return rett
            op.expr_type = left
            return left
        elif isinstance(op.op, Mult):
            left = op.left.accept(self)
            right = op.right.accept(self)
            # HACK: This should really be with a vector/matrix, 
            # but we don't support them well right now
            if left.isPrimitive():
                # we should do more checking here
                op.expr_type = right
                return right
            if right.isPrimitive():
                op.expr_type = left
                return left
            op.expr_type = Treal()
            return Treal()
        elif isinstance(op.op, Div):
            left = op.left.accept(self)
            right = op.right.accept(self)
            # HACK: This should really be with a vector/matrix, 
            # but we don't support them well right now
            if left.isPrimitive():
                # we should do more checking here
                op.expr_type = right
                return right
            if right.isPrimitive():
                op.expr_type = left
                return left
            op.expr_type = Treal()
            return Treal()
        else:
            assert False, f"Type inference for operator {type(op.op)} is not yet supported"

    def visitUnaryOperator(self, op:BinaryOperator):
        if isinstance(op.op, (UPlus, UMinus)):
            value = op.value.accept(self)
            op.expr_type = value
            return value
        else:
            assert False, f"Type inference for operator {type(op.op)} is not yet supported"

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
