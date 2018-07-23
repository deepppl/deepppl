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

import ast
import torch
import astpretty
import astor
import sys
from .ir import NetVariable, Program, ForStmt, ConditionalStmt, \
                AssignStmt, Subscript, BlockStmt,\
                CallStmt, List, SamplingDeclaration, SamplingObserved,\
                SamplingParameters, Variable

from .exceptions import MissingPriorNetException, MissingGuideNetException,\
                         MissingModelExeption, MissingGuideExeption, \
                        ObserveOnGuideExeption, UnsupportedProperty, \
                        UndeclaredParametersException, UndeclaredNetworkException,\
                        InvalidSamplingException, UndeclaredVariableException,\
                        UnknownDistributionException

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
    def __init__(self):
        super(VariableAnnotationsVisitor, self).__init__()
        self.ctx = {}
        self.block = None

    def defaultVisit(self, node):
        answer = node
        answer.children = self._visitChildren(node)
        return answer

    def _addVariable(self, name):
        if name in self.ctx:
            assert False, "Variable: {} already declared.".format(name)
        self.ctx[name] = self.block

    def _delVariable(self, name):
        if not name in self.ctx:
            assert False, "Trying to delete an inexistent variable:{}.".format(name)
        del self.ctx[name]
    
    def visitProgram(self, program):
        answer = Program()
        answer.children = self._visitChildren(program)
        return answer

    def visitForStmt(self, forstmt):
        id = forstmt.id
        answer = ForStmt(id = id)
        self._addVariable(id)
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
        if (self.block.is_prior() or self.block.is_guide()) and target.is_net_var():
            method = SamplingDeclaration
        elif target.is_data_var():
            method = SamplingObserved
        elif target.is_params_var():
            method = SamplingParameters
        else:
            assert False, "Don't know how to sample this:{}".format(sampling)
        return method(target = target, args = args, id = sampling.id)

    def visitProgramBlock(self, block):
        self.block = block
        return self.defaultVisit(block)

    visitData = visitProgramBlock
    visitModel = visitProgramBlock
    visitParameters = visitProgramBlock
    visitGuide = visitProgramBlock
    visitGuideParameters = visitProgramBlock
    visitPrior = visitProgramBlock
        
    def visitVariableDecl(self, decl):
        decl.dim = decl.dim.accept(self) if decl.dim else None
        name = decl.id
        self._addVariable(name)
        return decl

    def visitVariable(self, var):
        name = var.id
        if name not in self.ctx:
            raise UndeclaredVariableException(name)
        var.block_name = self.ctx[name].blockName()
        return var


class NetworkVisitor(IRVisitor):
    def __init__(self):
        super(NetworkVisitor, self).__init__()
        self._nets = {}
        self._priors = {}
        self._guides = {}
        self._shouldRemove = False

    def defaultVisit(self, node):
        answer = node
        answer.children = self._visitChildren(node)
        return answer

    def visitProgram(self, program):
        answer = Program()
        answer.children = self._visitChildren(program)
        return answer

    def visitSamplingBlock(self, sampling):
        self._shouldRemove = True
        answer = self.defaultVisit(sampling)
        self._shouldRemove = False
        return answer

    visitSamplingObserved = visitSamplingBlock
    visitSamplingDeclaration = visitSamplingBlock
    visitSamplingParameters = visitSamplingBlock

    def visitNetVariable(self, var):
        net = var.name
        params = var.ids
        if not net in self._nets:
            raise UndeclaredNetworkException(net)
        if not params in self._nets[net].params:
            raise UndeclaredParametersException('.'.join([net] + params))
        if self._shouldRemove:
            self._currdict[net].remove(self._param_to_name(params))
        return var

    def visitPrior(self, prior):
        self._currdict = self._priors
        self._currBlock = prior
        nets = [net for net in self._currdict if self._currdict[net]]
        prior.children = self._visitChildren(prior)
        prior._nets = nets
        for net in self._currdict:
            params = self._currdict[net]
            if params:
                raise MissingPriorNetException(net, params)
        self._currdict = None
        self._currBlock = None
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


    def visitGuide(self, guide):
        self._currdict = self._guides
        self._currBlock = guide
        nets = [net for net in self._currdict if self._currdict[net]]
        guide.children = self._visitChildren(guide)
        guide._nets = nets
        for net in self._currdict:
            params = self._currdict[net]
            if params:
                raise MissingGuideNetException(net, params)
        self._currdict = None
        self._currBlock = None
        return guide
        
    def _param_to_name(self, params):
        return '.'.join(params)

    def visitNetDeclaration(self, decl):
        name = decl.name
        self._nets[name] = decl
        if decl.params:
            self._guides[name] = set([self._param_to_name(x) for x in decl.params])
            self._priors[name] = set([self._param_to_name(x) for x in decl.params])
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
        answer = node
        answer.children = self._visitChildren(node)
        return answer

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
        self.compareToOrRaise(answer.model, MissingModelExeption)
        self.compareToOrRaise(answer.guide, MissingGuideExeption)
        return answer

    def visitSamplingBlock(self, block):
        self._currentBlock = block
        answer = self.defaultVisit(block)
        self._currentBlock = None
        return answer

    visitGuide = visitSamplingBlock
    visitPrior = visitSamplingBlock
    visitModel = visitSamplingBlock

    def visitSamplingParameters(self, sampling):
        if self._currentBlock is not None:
            if isinstance(sampling.target, Variable):
                id = sampling.target.id
            elif isinstance(sampling.target, Subscript): 
                ## XXX A more general logic must be applied elsewhere
                id = sampling.target.id.id
            else:
                raise InvalidSamplingException(sampling.target)
            self._currentBlock.addSampled(id)
        return self.visitSamplingStmt(sampling)

    def visitSamplingObserved(self, obs):
        if self._currentBlock and self._currentBlock.is_guide():
            raise ObserveOnGuideExeption(obs.target.id)
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

class Ir2PythonVisitor(IRVisitor):
    def __init__(self):
        super(Ir2PythonVisitor, self).__init__()
        self.data_names = set()
        self._priors = {}
        self.target_name_visitor = TargetVisitor(self)
        self.helper = PythonASTHelper()
        self._model_header = []
        self._guide_header = []

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

    def _assign(self, target_node, value):
        if isinstance(target_node, ast.Name):
            ## XXX instead of loadName, storeName should be used
            target = ast.Name(id = target_node.id, ctx = ast.Store())
        elif isinstance(target_node, ast.Subscript):
            target = ast.Subscript(value = target_node.value,
                                  slice = target_node.slice,
                                  ctx = ast.Store())
        else:
            assert False, "Don't know how to assign to {}".format(target_node)
        return ast.Assign(
                targets=[target],
                value = value)

    def _call(self, id_, args_):
        ## TODO id is a string!
        id = self.loadName(id_)
        args = args_.accept(self)
        args = args.elts if args else []
        return self.call(id, args=args)

    def _funcDef(self, name = None, args = [], body = []):
        return ast.FunctionDef(
                name = name,
                args = ast.arguments(args = args,
                                    vararg = None,
                                    kwonlyargs = [],
                                    kw_defaults=[],
                                    kwarg=None,
                                    defaults=[]),
                body = body,
                decorator_list = [],
                returns = None
            )

    def targetToName(self, target):
        if isinstance(target, ast.Name):
            return ast.Str(target.id)
        elif isinstance(target, ast.Subscript):
            base = self.targetToName(target.value)
            arg = target.slice.value
            format = self.loadAttr(ast.Str('{}'), 'format')
            formatted = self.call(format, args = [arg,])
            return ast.BinOp(left = base,
                             right = formatted,
                             op = ast.Add())
        else:
            assert False, "Don't know how to stringfy: {}".format(target)

    def visitConstant(self, const):
        tensor = self.loadName('tensor')
        args = [ast.Num(const.value)]
        return self.call(tensor, args = args)

    def visitVariableDecl(self, decl):
        if decl.data:
            self.data_names.add(decl.id)
        dims = decl.dim.accept(self) if decl.dim else None
        if dims:
            ## XXX we are ignoring the initialization.
            shapes = ast.Subscript(
                                    value = self.loadName('___shape'),
                                    slice = ast.Index(value = ast.Str(decl.id)),
                                    ctx = ast.Store())

            return self._assign(shapes, dims)
        if decl.init:
            ## XXX
            if True:
                raise NotImplementedError
            assert not decl.data, "Data cannot be initialized"
            init = decl.init.accept(self)
            torch_ = self.loadName('torch')
            zeros_fn = self.loadAttr(torch_, 'zeros')
            dims = [ast.List(elts=dims, ctx=ast.Load())]
            zeros = self.call(zeros_fn, args = dims)
            return ast.Assign(
                    targets=[ast.Name(id=decl.id, ctx=ast.Store())],
                    value=zeros)
        return None


    def visitList(self, list):
        elts = self._visitAll(list.elements)
        return ast.List(elts= elts,
                        ctx = ast.Load())

    def visitVariable(self, var):
        assert var.block_name is not None
        return self.loadName(var.id)

    def visitCallStmt(self, call):
        return self._call(call.id, call.args)

    def visitAssignStmt(self, ir):
        target, value = self._visitChildren(ir)
        if ir.target.is_variable():
            if ir.target.is_guide_parameters_var():
                target_name = self.targetToName(target)
                value = self.call(self._pyroattr('param'),
                                args = [
                                        target_name,
                                        value,
                                        ## XXX possible constraints
                                ])
        return self._assign(target, value)


    def _pyroattr(self, attr):
        return self.loadAttr(self.loadName('pyro'), attr)

    def visitForStmt(self, forstmt):
        ## TODO: id is not an object of ir!
        id = forstmt.id
        from_, to_, body = self._visitChildren(forstmt)
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

    def visitPlus(self, dummy):
        return ast.Add()

    def visitMinus(self, dummy):
        return ast.Sub()

    def visitMult(self, dummy):
        return ast.Mult()

    def visitDiv(self, dummy):
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

    def visitSubscript(self, subscript):
        id, idx = self._visitChildren(subscript)
        idx_z = ast.BinOp(left = idx, right = ast.Num(1), op = ast.Sub())
        return ast.Subscript(
                value = id,
                slice=ast.Index(value=idx_z),
                ctx=ast.Load())

    def samplingDist(self, sampling):
        args = [arg.accept(self) for arg in sampling.args]
        id = sampling.id
        if hasattr(torch.distributions, id):
            # Check if the distribution exists in torch.distributions
            dist = self.loadAttr(self.loadName('dist'), id)
        ## XXX We need keyword parameters
        elif id.lower() == 'CategoricalLogits'.lower():
            dist = self.loadName('CategoricalLogits')
        else:
            raise UnknownDistributionException(id)
        return self.call(dist,
                        args = args)
        

    def visitSamplingDeclaration(self, sampling):
        """This node represents when a variable is declared to have a given distribution"""
        target = sampling.target.accept(self)
        dist = self.samplingDist(sampling)
        return self._assign(target, dist)

    def samplingCall(self, sampling, target, keywords = []):
        dist = self.samplingDist(sampling)
        sample = self._pyroattr('sample')
        target_name = self.targetToName(target)
        return self.call(sample,
                        args = [target_name, dist],
                        keywords=keywords)

    def visitSamplingObserved(self, sampling):
        """Sample statement on data."""
        target = sampling.target.accept(self)
        keyword = ast.keyword(arg='obs', value = target)
        call = self.samplingCall(sampling, target, keywords = [keyword])
        return ast.Expr(value = call)

    def visitSamplingParameters(self, sampling):
        """Sample a parameter."""
        target = sampling.target.accept(self)
        call = self.samplingCall(sampling, target)
        return self._assign(target, call)

    def visitNetVariable(self, var):
        name = self.loadName('{}_{}'.format(var.block_name, var.name))
        value = '.'.join(var.ids)  ##XXX
        return ast.Subscript(
                value = name,
                slice = ast.Index(value = ast.Str(value)),
                ctx = ast.Load()
        )

    def visitVariableProperty(self, prop):
        var = ast.Str(prop.var.id)
        if prop.prop != 'shape':
            raise UnsupportedProperty(prop)
        dict_ = self.loadName('___' + prop.prop)
        return ast.Subscript(
            value = dict_,
            slice = ast.Index(value = var),
            ctx = ast.Load()
        )

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

    visitParameters = visitData
    visitGuideParameters = visitParameters

    def visitNetworksBlock(self, netBlock):
        return None

    def visitModel(self, model):
        body = self.liftBlackBox(model)
        body.extend(self._visitChildren(model))
        body = self._ensureStmtList(body)
        return self.buildModel(body)

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
        ## TODO: only one nn is suported in here.
        pre_body = self.liftBlackBox(prior)
        inner_body = self.liftBody(prior, name, name_prior)
        body = self._model_header + pre_body + inner_body
        f = self._funcDef(name = name_prior, args = self.modelArgs(), body = body)
        self._priors = {name : name_prior}
        return f

    def modelArgsAsParams(self):
        return [self.loadName(name) for name in sorted(self.data_names)]


    def visitGuide(self, guide):
        is_net = len(guide._nets) > 0
        assert (is_net and len(guide._nets) == 1) or not is_net
        name = guide._nets[0] if is_net else  ''
        ## TODO: only one nn is suported in here.
        name_guide = 'guide_' + name ## XXX
        pre_body = self.liftBlackBox(guide)

        if is_net:
            inner_body = self.liftBody(guide, name, name_guide)
        else:
            inner_body = [x for x in self._visitChildren(guide) if x is not None]
        body = self._guide_header + pre_body + inner_body

        f = self._funcDef(name = name_guide, 
                            args = self.modelArgs(), 
                            body = body)
        return f

    def modelArgs(self):
        return [ast.arg(name, None) for name in sorted(self.data_names)]

    def buildModel(self, inner_body):
        pre_body = []
        for prior in self._priors:
            pre_body.append(
                self._assign(self.loadName(prior),
                             self.call(
                                        id = self.loadName(self._priors[prior]),
                                        args = self.modelArgsAsParams()
                                        )
                            )
            )
        body = self._model_header + pre_body + inner_body
        model = self._funcDef(
                            name = 'model',
                            args = self.modelArgs(),
                            body = body)
        return model

    def buildBasicHeaders(self):
        name = self.storeName('___shape')
        sizes = self._assign(name, ast.Dict(keys=[], values=[]))
        return [sizes,]

    def buildHeaders(self, program):
        transform = lambda block: block.accept(self) if block else []
        basic = self.buildBasicHeaders()
        self._model_header = basic + transform(program.data) + transform(program.parameters)
        self._guide_header = self._model_header + transform(program.guideparameters)

    def visitProgram(self, program):
        self.buildHeaders(program)
        python_nodes = [node.accept(self) for node in [
                                                        program.guide,
                                                        program.prior,
                                                        program.model]
                                                if node]
        body = self._ensureStmtList(python_nodes)
        module = ast.Module()
        module.body = [
            self.import_('torch'),
            self.importFrom_('torch', ['tensor',]),
            self.import_('pyro'),
            self.import_('pyro.distributions', 'dist')]
        module.body += body
        ast.fix_missing_locations(module)
        if from_test():
            astpretty.pprint(module)
            print(astor.to_source(module))
        return module

            


def ir2python(ir):
    annotator = VariableAnnotationsVisitor()
    ir = ir.accept(annotator)
    nets = NetworkVisitor()
    ir = ir.accept(nets)
    consistency = SamplingConsistencyVisitor()
    ir = ir.accept(consistency)
    visitor = Ir2PythonVisitor()
    return ir.accept(visitor)



                                    