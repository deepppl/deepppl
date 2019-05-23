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
from contextlib import contextmanager
import ast
import torch
import astpretty
import astor
import sys

from .sdim import KnownDimension, makeGroupCanonLookup
from .ir import NetVariable, Program, ForStmt, ConditionalStmt, \
                AssignStmt, Subscript, BlockStmt,\
                CallStmt, List, SamplingStmt, SamplingDeclaration, SamplingObserved,\
                SamplingParameters, Variable, Constant, BinaryOperator, \
                Minus, UnaryOperator, UMinus, AnonymousShapeProperty,\
                VariableProperty, NetVariableProperty, Prior, \
                TransformedParameters, Model

from .exceptions import *
from .stype_infer import TypeInferenceVisitor

from_test = lambda: hasattr(sys, "_called_from_test")

class IRVisitor(object):
    def defaultVisit(self, node):
        raise NotImplementedError

    def _defaultVisit(self, node):
        answer = node
        # Hack: what is a better way to do this?
        if isinstance(node, SamplingStmt):
            answer.args = self._visitAll(node.args)
            if node.target is not None:
                answer.target = node.target.accept(self)
            if node.shape is not None:
                answer.shape = node.shape.accept(self)
        else:
            answer.children = self._visitChildren(node)
        return answer

    def __getattr__(self, attr):
        if attr.startswith('visit'):
            return self.defaultVisit
        return self.__getattribute__(attr)

    def _visitAll(self, iterable):
        filter = lambda x: (x is not None) or None
        return [filter(x) and x.accept(self) for x in iterable ]

    def _visitChildren(self, node):
        return self._visitAll(node.children)

class TargetVisitor(IRVisitor):
    def __init__(self, ir2py):
        super(TargetVisitor, self).__init__()
        self.ir2py = ir2py

    def visitVariable(self, var):
        return var.id

    def visitNetVariable(self, var):
        return var.name + '.'.join(var.ids)

    def visitSubscript(self, subs):
        return subs.id.accept(self)


class VariableAnnotationsVisitor(IRVisitor):
    """Annotate all variables with their declaration's block.
    Additionally, replace each ~ with their corresponding object
    as all the information is already available."""
    def __init__(self):
        super(VariableAnnotationsVisitor, self).__init__()
        self.ctx = {}
        self.block2decl = defaultdict(list)
        self.block = None
        self._to_model = []

    def defaultVisit(self, node):
        return self._defaultVisit(node)

    def _addVariable(self, name, decl):
        if name in self.ctx:
            if self.ctx[name].blockName() == TransformedParameters.blockName():
                self.block2decl[self.block.blockName()].append(decl)
            else:
                raise AlreadyDeclaredException(name)
        else:
            self.ctx[name] = self.block
            self.block2decl[self.block.blockName()].append(decl)

    def _delVariable(self, name):
        if not name in self.ctx:
            assert False, "Trying to delete an nonexistent variable:{}.".format(name)
        del self.ctx[name]

    def visitProgram(self, program):
        answer = Program()
        answer.children = self._visitChildren(program)
        return answer

    def visitForStmt(self, forstmt):
        id = forstmt.id
        answer = ForStmt(id = id)
        self._addVariable(id, forstmt)
        answer.children = self._visitChildren(forstmt)
        self._delVariable(id)
        return answer

    def visitNetVariable(self, netvar):
        answer = NetVariable(name = netvar.name, ids = netvar.ids)
        answer.block_name = self.block.blockName()
        return answer

    def visitSamplingStmt(self, sampling):
        target = sampling.target.accept(self)
        args = self._visitAll(sampling.args)
        if self.block.is_guide():
            if target.is_net_var():
                method = SamplingDeclaration
            else:
                method = SamplingParameters
        else:
            method = SamplingObserved
        return method(target = target, args = args, id = sampling.id)

    def visitProgramBlock(self, block):
        self.block = block
        return self.defaultVisit(block)

    visitData = visitProgramBlock
    visitTransformedData = visitProgramBlock
    visitTransformedParameters = visitProgramBlock
    visitGuide = visitProgramBlock
    visitGuideParameters = visitProgramBlock
    visitPrior = visitProgramBlock
    visitGeneratedQuantities = visitProgramBlock

    def visitModel(self, model):
        model.body = self._to_model + model.body
        return self.visitProgramBlock(model)

    def visitParameters(self, params):
        block = self.visitProgramBlock(params)
        self._to_model = []
        for decl in self.block2decl[block.blockName()]:
            sampling = self._buildSamplingFor(decl)
            self._to_model.append(sampling)
        return block

    def visitSamplingFactor(self, sampling):
        """
        Implemented using an exponential distribution.
        This behavior is achieved using the following identity:
        ```
            target += exp   ==   -exp ~ exponential(1)
        ```
        """
        target = UnaryOperator(value = sampling.target, op = UMinus())
        args = [Constant(1.0),]
        observed = SamplingObserved(target = target,
                                    id = 'Exponential',
                                    args = args)
        return self.defaultVisit(observed)

    def _buildSamplingFor(self, decl):
        target = Variable(id = decl.id)
        target = target.accept(self)
        #XXX check constraints
        constraints = decl.type_.constraints
        if constraints:
            seen = {}
            for const in constraints:
                seen[const.sort] = const.value
            if 'lower' in seen and 'upper' in seen:
                dist = 'Uniform'
                args =  [seen['lower'], seen['upper']]
            elif 'lower' in seen:
                dist = 'LowerConstrainedImproperUniform'
                args = [seen['lower'],]
            elif 'upper' in seen:
                dist = 'UpperConstrainedImproperUniform'
                args = [seen['upper'],]
            else:
                assert False, 'unknown constraints: {}.'.format(seen)
        else:
            dist = 'ImproperUniform'
            args = []

        shape = decl.dim

        #XXX check dimensions
        sampling = SamplingParameters(
                        target = target,
                        args = args,
                        shape = shape,
                        id = dist)
        return sampling

    def visitVariableDecl(self, decl):
        decl.dim = decl.dim.accept(self) if decl.dim else None
        name = decl.id
        self._addVariable(name, decl)
        return decl

    def visitVariable(self, var):
        name = var.id
        if name not in self.ctx:
            raise UndeclaredVariableException(name)
        var.block_name = self.ctx[name].blockName()
        return var


class NetworksVisitor(IRVisitor):
    def __init__(self):
        super(NetworksVisitor, self).__init__()
        self._nets = {}
        self._priors = {}
        self._guides = {}
        self._shouldRemove = False
        self._shouldAddNetPrior = False

    def defaultVisit(self, node):
        return self._defaultVisit(node)

    def visitProgram(self, program):
        answer = Program()
        answer.children = self._visitChildren(program)
        if self._shouldAddNetPrior:
            prior = self.netPrior()
            answer.prior = prior.accept(self)
        return answer

    def visitSamplingBlock(self, sampling):
        self._shouldRemove = True
        answer = self.defaultVisit(sampling)
        self._shouldRemove = False
        return answer

    visitSamplingObserved = visitSamplingBlock
    visitSamplingDeclaration = visitSamplingBlock
    visitSamplingParameters = visitSamplingBlock

    def is_on_model(self):
        return self._currBlock and self._currBlock.is_model()

    def visitNetVariable(self, var):
        net = var.name
        params = var.ids
        if not net in self._nets:
            raise UndeclaredNetworkException(net)
        if not params in self._nets[net].params:
            raise UndeclaredParametersException('.'.join(params))
        if self._shouldRemove and not self.is_on_model():
            del self._currdict[net][var.id]
        return var

    def netPrior(self):
        body = []
        prior = Prior(body = body)
        for net in self._priors:
            for var in self._priors[net].values():
                shape = NetVariableProperty(var = var, prop = 'shape')
                sampling = SamplingDeclaration(
                            target = var,
                            id  = 'ImproperUniform',
                            args = [],
                            shape = shape
                        )
                var.block_name = prior.blockName()
                body.append(sampling)
        return prior

    def visitModel(self, model):
        self._currBlock = model
        model.children = self._visitChildren(model)
        self._currBlock = None
        return model

    def visitCallStmt(self, call):
        if call.id in self._nets.keys():
            name = call.id
            assert self._currBlock, "Use of NN in an unknown block."
            if len(self._nets[name].params) == 0:
                self._currBlock._blackBoxNets.add(call.id)
        return call

    def visitPrior(self, prior):
        return self._visitGenerativeBlock(prior, self._priors, MissingPriorNetException)

    def visitGuide(self, guide):
        return self._visitGenerativeBlock(guide, self._guides, MissingGuideNetException)

    def _visitGenerativeBlock(self, block, dict, exception):
        self._currdict = dict
        self._currBlock = block
        nets = [net for net in self._currdict if self._currdict[net]]
        block.children = self._visitChildren(block)
        block._nets = nets
        for net in self._currdict:
            params = self._currdict[net]
            if params:
                raise exception(net, params)
        self._currdict = None
        self._currBlock = None
        return block

    def _param_to_name(self, params):
        return '.'.join(params)

    def visitNetDeclaration(self, decl):
        name = decl.name
        self._nets[name] = decl
        if decl.params:
            vars = [NetVariable(name = name, ids = x)
                                            for x in decl.params]
            self._guides[name] = {var.id: var for var in vars}
            self._priors[name] = {var.id: var for var in vars}
            self._shouldAddNetPrior = True
        return decl


class SamplingConsistencyVisitor(IRVisitor):
    """For SVI, there are a couple of rules that must be satisfied:
        1. `parameters` block defines the latent variables.
        2. all latents must be sampled both in the `guide` and the `model`.
        3. `data` cannot be observed inside the `guide`."""
    def __init__(self):
        super(SamplingConsistencyVisitor, self).__init__()
        self._latents = set()
        self._currentBlock = None
        self._declarations = None

    def defaultVisit(self, node):
        return self._defaultVisit(node)

    def visitVariableDecl(self, var):
        if self._declarations is not None:
            self._declarations.add(var.id)
        return self.defaultVisit(var)

    def visitParameters(self, parameters):
        self._declarations = self._latents
        answer = self.defaultVisit(parameters)
        self._declarations = None
        return answer

    def compareToOrRaise(self, block, exception):
        if block:
            diff = self._latents.difference(block._sampled)
            if diff:
                raise exception(diff)

    def visitProgram(self, program):
        answer = Program()
        answer.children = self._visitChildren(program)
        self.compareToOrRaise(answer.model, MissingModelException)
        self.compareToOrRaise(answer.guide, MissingGuideException)
        return answer

    def visitSamplingBlock(self, block):
        old = self._currentBlock
        self._currentBlock = block
        answer = self.defaultVisit(block)
        self._currentBlock = old
        return answer

    visitModel = visitSamplingBlock
    visitGuide = visitSamplingBlock
    visitPrior = visitSamplingBlock

    def visitSamplingParameters(self, sampling):
        if self._currentBlock is not None:
            target = sampling.target
            if isinstance(target, Variable):
                id = target.id
            elif isinstance(target, Subscript):
                ## XXX A more general logic must be applied elsewhere
                id = target.id.id
            else:
                raise InvalidSamplingException(sampling.target)
            self._checkSamplingInGuide(target)
            self._currentBlock.addSampled(id)
        return self.visitSamplingStmt(sampling)

    def _checkSamplingInGuide(self, target):
        if not target.is_params_var():
            raise ObserveOnGuideException(target.id)


    def visitSamplingObserved(self, obs):
        if self._currentBlock and self._currentBlock.is_guide():
            raise ObserveOnGuideException(obs.target.id)
        return self.visitSamplingStmt(obs)


"Helper class for common `ast` objects"
class PythonASTHelper(object):
    def ensureStmt(self, node):
        if isinstance(node, ast.stmt):
            return node
        else:
            return ast.Expr(node)

    def ensureStmtList(self, listOrNode):
        if type(listOrNode) == type([]):
            return [self.ensureStmt(x) for x in listOrNode if x is not None]
        else:
            return [self.ensureStmt(listOrNode)]

    def call(self, func, args = [], keywords = []):
        return ast.Call(func = func,
                        args=args,
                        keywords=keywords)

    def loadName(self, name):
        return ast.Name(id = name, ctx = ast.Load())

    def storeName(self, name):
        return ast.Name(id = name, ctx = ast.Store())

    def loadAttr(self, obj, attr):
        return ast.Attribute(value = obj, attr = attr, ctx = ast.Load())

    def import_(self, name, asname = None):
        return ast.Import(names=[ast.alias(name=name, asname=asname)])

    def importFrom_(self, module, names):
        names_ = [ast.alias(name=name, asname = None) for name in names]
        return ast.ImportFrom(module, names_, 0)

class VariableInitializationVisitor(IRVisitor):
    def __init__(self):
        super(VariableInitializationVisitor, self).__init__()
        self._currentBlock = None
        self._to_guide = []
        self._emitNext = None
        self._variationalParameters = OrderedDict()

    @contextmanager
    def _inBlock(self, block):
        try:
            old = self._currentBlock
            self._currentBlock = block
            yield
        finally:
            self._currentBlock = old

    def _visitBlock(self, block):
        with self._inBlock(block):
            return self.defaultVisit(block)

    def _visitAll(self, iterable):
        filter = lambda x: (x is not None) or None
        answer = []
        for x in iterable:
            answer.append(filter(x) and x.accept(self))
            if self._emitNext is not None:
                _next = self._emitNext
                self._emitNext = None
                answer.append(_next.accept(self))
        return answer


    visitData = _visitBlock
    visitTransformedData = _visitBlock
    visitParameters = _visitBlock
    visitTransformedParameters = _visitBlock
    visitGuideParameters = _visitBlock
    visitModel = _visitBlock
    visitGeneratedQuantities = _visitBlock

    def visitGuide(self, guide):
        with self._inBlock(guide):
            guide.body = self._to_guide + guide.body
            answer = self.defaultVisit(guide)
            defaultInits = self._buildDefaultInits()
            answer.body = defaultInits + answer.body
            return answer

    def _buildDefaultInits(self):
        answer = []
        for name in list(self._variationalParameters):
            target = Variable(id=name)
            shape = VariableProperty(var=target, prop='shape')
            args = List(elements=[shape, ])
            value = CallStmt(id='rand', args=args)
            constraints = self._variationalParameters[name].type_.constraints or []
            assign = AssignStmt(target = target, value = value, constraints = constraints).accept(self)
            answer.append(assign)
        return answer


    def visitAssignStmt(self, assign):
        if self._currentBlock.is_guide():
            if assign.target.is_variable():
                id = assign.target.id
                if id in self._variationalParameters:
                    del self._variationalParameters[id]
        return self.defaultVisit(assign)


    def visitVariableDecl(self, decl):
        answer = self.defaultVisit(decl)
        assert self._currentBlock is not None, "declaration outside a block."
        if self._currentBlock.is_guide_parameters():
            self._variationalParameters[decl.id] = decl
        if decl.init:
            if self._currentBlock.is_data() or self._currentBlock.is_parameters():
                assert False, "Initialization of data or parameters is forbiden."
            initialized = self._buildInit(answer)
            if not self._currentBlock.is_transformed_parameters():
                answer.init = None
            if self._currentBlock.is_guide_parameters():
                self._to_guide.append(initialized)
            else:
                self._emitNext = initialized
        else:
            if (self._currentBlock.is_data()
                or self._currentBlock.is_parameters()
                or self._currentBlock.is_guide_parameters()
                or self._currentBlock.is_guide()):
                pass
            else:
                self._emitNext = self._buildDefaultArray(answer)
        return answer

    def _buildInit(self, decl):
        target = Variable(id = decl.id)
        value = decl.init
        return AssignStmt(target = target, value = value)

    def _buildDefaultArray(self, decl):
        if decl.type_.is_array or decl.dim is not None:
            target = Variable(id = decl.id)
            shape = VariableProperty(var = target, prop = 'shape')
            args = List(elements = [shape,])
            value = CallStmt(id = 'zeros', args = args) # XXX This value should depend on the type of the variable
            return AssignStmt(target = target, value = value)
        else:
            return None

    def defaultVisit(self, node):
        return self._defaultVisit(node)

    def visitProgram(self, program):
        answer = Program()
        answer.children = self._visitChildren(program)
        return answer


class Ir2PythonVisitor(IRVisitor):
    new_distributions = {name.lower(): name for name in [
                            'bernoulli_logit',
                            'categorical_logits',
                            'binomial_logit',
                            'poisson_log',
                            'ImproperUniform',
                            'LowerConstrainedImproperUniform',
                            'UpperConstrainedImproperUniform',
    ]}
    renamed_distributions = {
        'multi_normal': 'MultivariateNormal',
        'logistic': 'LogisticNormal',
        'student_t' : 'StudentT'
    }



    def __init__(self, type_infer, verbose=False):
        super(Ir2PythonVisitor, self).__init__()
        self._program = None
        self.data_names = set()
        self._transformed_data_names = set()
        self._parameters_names = set()
        self._transformed_parameters_names = set()
        self._transformed_parameters = []
        self._generated_quantities_names = set()
        self._priors = {}
        self.target_name_visitor = TargetVisitor(self)
        self.helper = PythonASTHelper()
        self._model_header = []
        self._guide_header = []
        self._observed = 0
        self.forIndexes = []
        self.type_infer = type_infer
        self.verbose = verbose

    def _ensureStmt(self, node):
        return self.helper.ensureStmt(node)

    def _ensureStmtList(self, listOrNode):
        return self.helper.ensureStmtList(listOrNode)

    def is_data(self, ir):
        target = ir.accept(self.target_name_visitor)
        return target in self.data_names

    def loadName(self, name):
        return self.helper.loadName(name)

    def storeName(self, name):
        return self.helper.storeName(name)

    def import_(self, name, asname = None):
        return self.helper.import_(name, asname = asname)

    def importFrom_(self, module, names):
        return self.helper.importFrom_(module, names)

    def loadAttr(self, obj, attr):
        return self.helper.loadAttr(obj, attr)

    def call(self, id,  args = [], keywords = []):
        return self.helper.call(id, args = args, keywords = keywords)

    def _assign(self, target_node, value, var_type=None):
        if isinstance(target_node, ast.Name):
            ## XXX instead of loadName, storeName should be used
            target = ast.Name(id = target_node.id, ctx = ast.Store())
        elif isinstance(target_node, ast.Subscript):
            target = ast.Subscript(value = target_node.value,
                                  slice = target_node.slice,
                                  ctx = ast.Store())
        else:
            assert False, "Don't know how to assign to {}".format(target_node)
        if var_type is None:
            return ast.Assign(
                    targets=[target],
                    value = value)
        else:
            return ast.AnnAssign(
                    target=target,
                    annotation=var_type,
                    simple=1,
                    value = value)

    def _call(self, id_, args_):
        ## TODO id is a string!
        id = self.loadName(id_)
        args = args_.accept(self)
        args = args.elts if args else []
        return self.call(id, args=args)

    def _funcDef(self, name = None, args = [], defaults=[], body = []):
        return ast.FunctionDef(
                name = name,
                args = ast.arguments(args = args,
                                    vararg = None,
                                    kwonlyargs = [],
                                    kw_defaults=[],
                                    kwarg=None,
                                    defaults=defaults),
                body = body,
                decorator_list = [],
                returns = None
            )

    def targetToName(self, target, observed = None):
        if isinstance(target, ast.Name):
            base = ast.Str(target.id)
        elif isinstance(target, ast.Subscript):
            base = self.targetToName(target.value)
            arg = target.slice.value
            format = self.loadAttr(ast.Str('__{}'), 'format')
            formatted = self.call(format, args = [arg,])
            base = ast.BinOp(left = base,
                             right = formatted,
                             op = ast.Add())
        elif observed is not None:
            # arbitrary expressions
            base = ast.Str('expr')
            for idx in self.forIndexes:
                arg = self.loadName(idx)
                format = self.loadAttr(ast.Str('__{}'), 'format')
                formatted = self.call(format, args = [arg,])
                base = ast.BinOp(left = base,
                                 right = formatted,
                                 op = ast.Add())
        else:
            assert False, "Don't know how to stringfy: {}".format(target)
        if observed is None:
            return base
        else:
            observed_str = ast.Str('__' + str(observed))
            return ast.BinOp(left = base, right = observed_str, op = ast.Add())

    def visitConstant(self, const):
        if hasattr(const, 'expr_type'):
            dims = const.expr_type.dimensions()
            if not dims:
                return ast.Num(const.value)
            args = [self.dimensionToAST(d) for d in dims]
            if const.value == 0:
                return self.call(self.loadName("zeros"), args=args)
            else:
                ones = self.call(self.loadName("ones"), args=args)
                if const.value == 1:
                    return ones
                else:
                    return ast.BinOp(left = ast.Num(const.value),
                            right = ones,
                            op = ast.Mult())
        return ast.Num(const.value)

    def visitVariableDecl(self, decl):
        if decl.data:
            self.data_names.add(decl.id)
        if decl.transformed_data:
            self._transformed_data_names.add(decl.id)
        if decl.parameters:
            self._parameters_names.add(decl.id)
        if decl.transformed_parameters:
            self._transformed_parameters_names.add(decl.id)
        if decl.generated_quantities:
            self._generated_quantities_names.add(decl.id)


    def visitList(self, list):
        elts = self._visitAll(list.elements)
        return ast.List(elts= elts,
                        ctx=ast.Load())

    def visitTuple(self, tuple):
        elts = self._visitAll(tuple.exprs)
        return ast.Tuple(elts=elts, ctx=ast.Load())

    def visitAnonymousShapeProperty(self, prop):
        pass

    def visitVariable(self, var):
        assert var.block_name is not None
        return self.loadName(var.id)

    def visitCallStmt(self, call):
        return self._call(call.id, call.args)

    def uniformBounds(self, value, lower, upper):
        x = ast.BinOp(
            left=upper,
            op=ast.Sub(),
            right=lower)
        return ast.BinOp(
            left=ast.BinOp(
                left=x,
                op=ast.Mult(),
                right=value),
            op=ast.Add(),
            right=lower)

    def visitAssignStmt(self, ir):
        target, value = self._visitChildren(ir)
        if ir.target.is_variable():
            if ir.target.is_guide_parameters_var():
                target_name = self.targetToName(target)
                if ir.constraints is not None:
                    seen = {}
                    for const in ir.constraints:
                        seen[const.sort] = self.visitConstant(const.value)
                    if 'lower' in seen and 'upper' in seen:
                        value = self.uniformBounds(value,
                            seen['lower'],
                            seen['upper'])
                    elif 'lower' in seen:
                        value = self.uniformBounds(value,
                            seen['lower'],
                            ast.BinOp(left=seen['lower'], op=ast.Add(), right=ast.Num(n=10)))
                    elif 'upper' in seen:
                        value = self.uniformBounds(value,
                            ast.BinOp(left=seen['upper'], op=ast.Sub(), right=ast.Num(n=10)),
                            seen['upper'])
                    else:
                        value = self.uniformBounds(value,
                            ast.UnaryOp(op=ast.USub(), operand=ast.Num(n=2)),
                            ast.Num(n=2))
                value = self.call(self._pyroattr('param'),
                            args = [
                                    target_name,
                                    value,
                                    ## XXX possible constraints
                            ])
        if self.verbose and ir.target.expr_type is not None:
            ctype = ir.target.expr_type.canon((self.type_infer.dims_canon_map))
            return self._assign(target, value, var_type=self.typeToAnnotation(ctype))
        else:
            return self._assign(target, value)

    def _pyroattr(self, attr):
        return self.loadAttr(self.loadName('pyro'), attr)

    def visitForStmt(self, forstmt):
        ## TODO: id is not an object of ir!
        id = forstmt.id
        self.forIndexes.append(id)
        from_, to_, body = self._visitChildren(forstmt)
        self.forIndexes.pop()
        incl_to = ast.BinOp(left = to_,
                            right = ast.Num(1),
                            op = ast.Add())
        interval = [from_, incl_to]
        iter = self.call(self.loadName('range'),
                                interval)
        body = self._ensureStmtList(body)
        return ast.For(target = ast.Name(id = id, ctx=ast.Store()),
                        iter = iter,
                        body = body,
                        orelse = [])

    def visitConditionalStmt(self, conditional):
        test, true = self._visitAll((conditional.test,
                                   conditional.true))
        false = conditional.false.accept(self) if conditional.false else []
        true, false = [self._ensureStmtList(x) for x in (true, false)]
        return ast.If(test=test, body=true, orelse=false)


    def visitBlockStmt(self, block):
        return self._visitAll(block.body)

    def visitBinaryOperator(self, binaryop):
        left, right, op = self._visitChildren(binaryop)
        if isinstance(op, ast.cmpop):
            return ast.Compare(left = left,
                               ops = [op],
                               comparators =[right])
        elif isinstance(op, ast.boolop):
            return ast.BoolOp(op = op, values = [left, right])
        return ast.BinOp(left = left, right = right, op = op)

    def visitUnaryOperator(self, unaryop):
        value, op = self._visitChildren(unaryop)
        return ast.UnaryOp(operand = value, op=op)

    def visitUPlus(self, dummy):
        return ast.UAdd()

    def visitUMinus(self, dummy):
        return ast.USub()

    def visitUNot(self, dummy):
        return ast.Not()

    def visitPlus(self, dummy):
        return ast.Add()

    def visitMinus(self, dummy):
        return ast.Sub()

    def visitMult(self, dummy):
        return ast.Mult()

    def visitDotMult(self, dummy):
        return ast.Mult()

    def visitDiv(self, dummy):
        return ast.Div()

    def visitDotDiv(self, dummy):
        return ast.Div()

    def visitAnd(self, dummy):
        return ast.And()

    def visitOr(self, dummy):
        return ast.Or()

    def visitLE(self, dummy):
        return ast.LtE()

    def visitGE(self, dummy):
        return ast.GtE()

    def visitLT(self, dummy):
        return ast.Lt()

    def visitGT(self, dummy):
        return ast.Gt()

    def visitEQ(self, dummy):
        return ast.Eq()

    def shiftIdx(self, idx):
        if isinstance(idx, ast.Tuple):
            elts = [ast.BinOp(left = x, right = ast.Num(1), op = ast.Sub()) for x in idx.elts]
            return ast.Tuple(elts=elts, ctx=ast.Load())
        else:
            return ast.BinOp(left = idx, right = ast.Num(1), op = ast.Sub())

    def visitSubscript(self, subscript):
        id, idx = self._visitChildren(subscript)
        idx_z = self.shiftIdx(idx)
        return ast.Subscript(
                value = id,
                slice=ast.Index(value=idx_z),
                ctx=ast.Load())

    def samplingDist(self, sampling):
        kwds = []
        result_shape = sampling.expr_type.dimensions()

        def fitShape(arg, id, argIndex, arg_type, target_shape):
            if target_shape:
                if id == 'normal' and argIndex > 0:
                    return arg
                arg_shape = arg_type.dimensions()
                arg_dims = len(arg_shape)
                if len(target_shape) > arg_dims:
                    target_dims = target_shape[arg_dims:]
                    dims = [self.dimensionToAST(d) for d in target_shape]
                    if isinstance(arg, ast.Num):
                        if arg.n == 0:
                            zeros = self.call(self.loadName("zeros"), args=dims)
                            return zeros
                        elif arg.n == 1:
                            ones = self.call(self.loadName("ones"), args=dims)
                            return zeros
                    ones = self.call(self.loadName("ones"), args=dims)
                    return ast.BinOp(left = arg,
                                right = ones,
                                op = ast.Mult())
                else:
                    return arg
            else:
                return arg

        args = [fitShape(arg.accept(self), sampling.id, i, arg.expr_type, result_shape) for (i, arg) in enumerate(sampling.args)]

        if sampling.shape:
            sh = sampling.shape.accept(self)
            if not isinstance(sh, ast.Tuple) or len(sh.elts) != 0:
                kwds.append(ast.keyword('shape', sh))
        id = sampling.id
        if hasattr(torch.distributions, id.capitalize()):
            # Check if the distribution exists in torch.distributions
            dist = self.loadAttr(self.loadName('dist'), id.capitalize())
        ## XXX We need keyword parameters
        elif id.lower() in self.renamed_distributions:
            dist = self.loadAttr(self.loadName('dist'), self.renamed_distributions[id.lower()])
        elif id.lower() in self.new_distributions:
            dist = self.loadName(self.new_distributions[id.lower()])
        else:
            raise UnknownDistributionException(id)
        return self.call(dist,
                        args = args,
                        keywords = kwds
                        )

    def visitSamplingDeclaration(self, sampling):
        """This node represents when a variable is declared to have a given distribution"""
        target = sampling.target.accept(self)
        dist = self.samplingDist(sampling)
        if self.verbose and sampling.expr_type is not None:
            ctype = sampling.expr_type.canon((self.type_infer.dims_canon_map))
            return self._assign(target, dist, var_type=self.typeToAnnotation(ctype))
        else:
            return self._assign(target, dist)

    def samplingCall(self, sampling, target, keywords = [], observed = None):
        dist = self.samplingDist(sampling)
        sample = self._pyroattr('sample')
        target_name = self.targetToName(target, observed = observed)
        return self.call(sample,
                        args = [target_name, dist],
                        keywords=keywords)

    def visitSamplingObserved(self, sampling):
        """Sample statement on data."""
        target = sampling.target.accept(self)
        keyword = ast.keyword(arg='obs', value = target)
        self._observed += 1
        call = self.samplingCall(sampling,
                                target,
                                keywords = [keyword],
                                observed = self._observed)
        return ast.Expr(value = call)

    def visitSamplingParameters(self, sampling):
        """Sample a parameter."""
        target = sampling.target.accept(self)
        call = self.samplingCall(sampling, target)
        if self.verbose and sampling.expr_type is not None:
            ctype = sampling.expr_type.canon((self.type_infer.dims_canon_map))
            return self._assign(target, call, var_type=self.typeToAnnotation(ctype))
        else:
            return self._assign(target, call)

    def visitNetVariable(self, var):
        name = self.loadName('{}_{}'.format(var.block_name, var.name))
        value = '.'.join(var.ids)  ##XXX
        return ast.Subscript(
                value = name,
                slice = ast.Index(value = ast.Str(value)),
                ctx = ast.Load()
        )

    def dimToAST(self, dim):
        if dim.isVariable():
            raise UnderspecifiedDimension(dim, "unknown")
        elif dim.isKnown():
            dd = dim.description()
            # TODO Avi: If this lookup fails
            # We probably want to verify that the original is a reasonable canonical element
            dcanon = self.type_infer.dims_canon_map.get(dd, dd)
            return self.knownDimensionToAST(dcanon)
        else:
            assert f"Unknown dimension {dim}"
    def visitVariableProperty(self, prop):
        if prop.prop != 'shape':
            raise UnsupportedProperty(prop)

        if hasattr(prop, 'expr_dim'):
            dims = prop.expr_dim
        else:
            dims = prop.var.expr_type.dimensions()
        if isinstance(dims, list):
            elts = [self.dimToAST(d) for d in dims]
            if len(elts) == 1:
                ast_dims = elts[0]
            else:
                ast_dims = ast.Tuple(elts = elts,
                                    ctx = ast.Load())
        else:
            ast_dims = self.dimToAST(dims)
        return ast_dims

    def dimensionToAST(self, d:'Dimension'):
        if d.isKnown():
            return self.knownDimensionToAST(d.description())
        else:
            raise UnderspecifiedDimension(d, "not concrete")

    def knownDimensionToAST(self, d:'KnownDimension'):
        a = ast.parse(d.expr(), mode='eval')
        if isinstance(a, ast.Expression):
            return a.body
        else:
            return a
        
    def pathAttr(self, path):
        paths = path.split('.')
        piter = iter(paths)
        p = self.loadName(next(piter))
        for attrib in piter:
            p = self.loadAttr(p, attrib)
        p = self.loadAttr(p, 'shape')
        return p

    def visitAnonymousShapeProperty(self, prop):
        return self.visitVariableProperty(prop)
        # dimension_id = prop.var.id
        # dim_dict = self.type_infer.denv
        # if dimension_id in dim_dict:
        #     d = dim_dict[dimension_id]
        #     return self.dimToAST(d)
        # else:
        #     raise UnderspecifiedDimension(dimension_id, "not found")

    def visitNetVariableProperty(self, netprop):
        net = netprop.var
        last = self.loadName(net.name)
        prop = netprop.prop
        if prop != 'shape':
            raise UnsupportedProperty(prop)
        for attr in net.ids + [prop]:
            last = self.loadAttr(last, attr)
        return last

    def visitNetDeclaration(self, decl):
        ## XXX do nothign
        assert False
        return None


    def visitData(self, data):
        ## TODO: implement data behavior
        answer = self._visitChildren(data)
        return self._ensureStmtList(answer)

    def visitTransformedData(self, transformed_data):
        name = 'transformed_data'
        args = self.modelArgs(no_transformed_data=True)
        body = []
        body.extend(self.buildBasicHeaders())
        body.extend(self._visitChildren(transformed_data))
        k = [ast.Str(td_name) for td_name in self._transformed_data_names]
        v = [self.loadName(td_name) for td_name in self._transformed_data_names]
        body.append(ast.Return(ast.Dict(k, v)))
        body = self._ensureStmtList(body)
        f = self._funcDef(name = name,
                          args = args['args'],
                          defaults = args['defaults'],
                          body = body)
        return f

    visitParameters = visitData

    def visitTransformedParameters(self, transformed_parameters):
        body = self._visitChildren(transformed_parameters)
        body = self._ensureStmtList(body)
        self._transformed_parameters = body
        return None

    visitGuideParameters = visitParameters

    def visitNetworksBlock(self, netBlock):
        return None

    def visitModel(self, model):
        body = self.liftBlackBox(model)
        body.extend(self._visitChildren(model))
        body = self._ensureStmtList(body)
        return self.buildModel(body)

    def getParametersSample(self, parameters, names):
        samples = []
        for name in sorted(names):
            # param = self.loadAttr(self.loadName(parameters), name)
            param = ast.Subscript(value = self.loadName(parameters),
                                  slice = ast.Index(value = ast.Str(name)),
                                  ctx = ast.Load())
            samples.append(self._assign(self.loadName(name), param))
        return samples

    def buildGeneratedQuantities(self):
        transformed_parameters = self._program.transformedparameters
        generated_quantities = self._program.generatedquantities
        name = 'generated_quantities'
        args = self.modelArgs(with_parameters_sample=True)
        body = []
        body.extend(self.buildTransformedDataAccess())
        body.extend(self.getParametersSample('parameters', self._parameters_names))
        body.extend(self.buildBasicHeaders())
        body.extend(self._transformed_parameters)
        k = [ast.Str(name) for name in self._transformed_parameters_names]
        v = [self.loadName(name) for name in self._transformed_parameters_names]
        if not (generated_quantities is None):
            body.extend(self._visitChildren(generated_quantities))
            k.extend([ast.Str(gq_name) for gq_name in sorted(self._generated_quantities_names)])
            v.extend([self.loadName(gq_name) for gq_name in sorted(self._generated_quantities_names)])
        body.append(ast.Return(ast.Dict(k, v)))
        body = self._ensureStmtList(body)
        f = self._funcDef(name = name,
                          args = args['args'],
                          defaults = args['defaults'],
                          body = body)
        return f

    def liftBlackBox(self, block):
        answer = []
        for name in block._blackBoxNets:
            mod = self._pyroattr('module')
            answer.append(self.call(
                                    mod,
                                    args = [
                                        ast.Str(name),
                                        self.loadName(name)
                                    ]))
        return self._ensureStmtList(answer)


    def liftModule(self, name, dict_name):
        rand_mod = self._pyroattr('random_module')
        lifted_name = 'lifted_' + name
        lifted = self.loadName(lifted_name)
        lifter = self.call(rand_mod,
                    args = [ast.Str(name),
                            self.loadName(name),
                            self.loadName(dict_name)])
        return [
                self._assign(lifted, lifter),
                ast.Return(value = self.call(lifted))]

    def liftBody(self, block, name, dict_name):
        inner_body = [x for x in self._visitChildren(block) if x is not None]
        body = [self._assign(self.loadName(dict_name),
                            ast.Dict(keys = [], values=[])),]
        body += inner_body
        body += self.liftModule(name, dict_name)
        return body

    def visitPrior(self, prior):
        is_net = len(prior._nets) > 0
        assert is_net and len(prior._nets) == 1
        name = prior._nets[0] if is_net else  ''
        name_prior = 'prior_' + name ## XXX
        args = self.modelArgs()
        ## TODO: only one nn is suported in here.
        pre_body = self.liftBlackBox(prior)
        inner_body = self.liftBody(prior, name, name_prior)
        body = self._model_header + pre_body + inner_body
        f = self._funcDef(name = name_prior, args = args['args'], defaults = args['defaults'], body = body)
        self._priors = {name : name_prior}
        return f

    def modelArgsAsParams(self, no_transformed_data=False):
        args = [self.loadName(name) for name in sorted(self.data_names)]
        if not no_transformed_data and self._transformed_data_names:
            td_args = modelArgsAsParams(self, no_transformed_data=True)
            td_call = self.call(
                id = self.loadName('transformed_data'),
                args = td_args
            )
            args.append(ast.arg(td_call))
        return [self.loadName(name) for name in sorted(self.data_names)]


    def visitGuide(self, guide):
        is_net = len(guide._nets) > 0
        assert (is_net and len(guide._nets) == 1) or not is_net
        name = guide._nets[0] if is_net else  ''
        args = self.modelArgs()
        ## TODO: only one nn is suported in here.
        name_guide = 'guide_' + name ## XXX
        pre_body = self.liftBlackBox(guide)
        td_access = self.buildTransformedDataAccess()

        if is_net:
            inner_body = self.liftBody(guide, name, name_guide)
        else:
            inner_body = [x for x in self._visitChildren(guide) if x is not None]
        body = self._guide_header + pre_body + inner_body

        f = self._funcDef(name = name_guide,
                            args = args['args'],
                            defaults = args['defaults'],
                            body = body)
        return f

    def typeOfVariable(self, var:str):
        t = self.type_infer.tenv.get(var, None)
        if t is None:
            return None
        else:
            return t.canon(self.type_infer.dims_canon_map)

    def typeToAnnotation(self, t):
        if t is None:
            return None
        else:
            return ast.Str(str(t))

    def modelArgs(self, no_transformed_data=False, with_parameters_sample=False):
        args = [ast.arg(name, self.typeToAnnotation(self.typeOfVariable(name)) if self.verbose else None)
            for name in sorted(self.data_names)]
        defaults = [ ast.NameConstant(None) for name in self.data_names ]
        if not no_transformed_data and self._transformed_data_names:
            args.append(ast.arg('transformed_data', None))
            defaults.append(ast.NameConstant(None))
        if with_parameters_sample:
            args.append(ast.arg('parameters', None))
            defaults.append(ast.NameConstant(None))
        return { 'args': args, 'defaults': defaults }

    def buildPrior(self, prior, basename):
        lifted_prior = self._assign(self.loadName(prior),
                             self.call(
                                        id = self.loadName(self._priors[prior]),
                                        args = self.modelArgsAsParams()
                                        )
                            )
        states = self.call(self.loadAttr(self.loadName(prior), 'named_parameters'))
        _dict = self.call(self.loadName('dict'), [states,])
        states_dict = self._assign(
                            self.loadName(f'{basename}_{prior}'),
                            _dict
                        )
        return [lifted_prior, states_dict]

    def buildTransformedDataAccess(self):
        td_access = []
        for td_name in self._transformed_data_names:
            access = ast.Subscript(value = self.loadName('transformed_data'),
                                   slice = ast.Index(value = ast.Str(td_name)),
                                   ctx = ast.Load())
            decl = self._assign(self.loadName(td_name),access)
            td_access.append(decl)
        return td_access

    def buildModel(self, inner_body):
        name = 'model'
        args = self.modelArgs()
        td_access = self.buildTransformedDataAccess()
        pre_body = []
        for prior in self._priors:
            pre_body.extend(self.buildPrior(prior, name))
        body = td_access + self._model_header + pre_body + inner_body
        model = self._funcDef(
                            name = name,
                            args = args['args'],
                            defaults = args['defaults'],
                            body = body)
        return model

    def buildBasicHeaders(self):
        return []

    def buildHeaders(self, program):
        transform = lambda block: block.accept(self) if block else []
        basic = self.buildBasicHeaders()
        tpd = transform(program.data)
        tpp = transform(program.parameters)
        self._model_header = basic + tpd + tpp
        tpgp = transform(program.guideparameters)
        self._guide_header = self._model_header + tpgp

    def visitProgram(self, program):
        self._program = program
        self.buildHeaders(program)
        python_nodes = [node.accept(self) for node in [
                                                        program.transformeddata,
                                                        program.transformedparameters,
                                                        program.guide,
                                                        program.prior,
                                                        program.model,]
                                                if node]
        if program.generatedquantities is not None or program.transformedparameters is not None:
            python_nodes += [self.buildGeneratedQuantities()]
        body = self._ensureStmtList(python_nodes)
        module = ast.Module()
        module.body = [
            self.import_('torch'),
            self.importFrom_('torch', ['tensor', 'rand']),
            self.import_('pyro'),
            self.import_('torch.distributions.constraints', 'constraints'),
            self.import_('pyro.distributions', 'dist')]
        module.body += body
        ast.fix_missing_locations(module)
        if from_test():
            # astpretty.pprint(module)
            print(astor.to_source(module))
        return module




def ir2python(ir, verbose=False):
    initialization = VariableInitializationVisitor()
    ir.accept(initialization)
    annotator = VariableAnnotationsVisitor()
    ir = ir.accept(annotator)
    nets = NetworksVisitor()
    ir = ir.accept(nets)
    consistency = SamplingConsistencyVisitor()
    ir = ir.accept(consistency)
    type_infer = TypeInferenceVisitor.run(ir)
    equality_grouper = type_infer.equalities
    equality_groups = equality_grouper.groups()
    type_infer.dims_canon_map = makeGroupCanonLookup(equality_groups)
    if verbose and len(equality_groups) != 0:
        print(f"WARNING: There are unproven equality constraints: {equality_grouper}.  The lookup map is {type_infer.dims_canon_map}")
    visitor = Ir2PythonVisitor(type_infer, verbose=verbose)
    a = ir.accept(visitor)
    ast.fix_missing_locations(a)
    return a




