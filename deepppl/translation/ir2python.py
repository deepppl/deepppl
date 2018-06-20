import ast
import torch
import astpretty
import astor
from .ir import NetVariable, Program, ForStmt, ConditionalStmt, \
                AssignStmt, SamplingStmt, Subscript, BlockStmt,\
                CallStmt, List

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
        return node

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
        answer.children = self._visitAll(program.children)
        return answer

    def visitAssignStmt(self, ir):
        answer = AssignStmt()
        answer.children = self._visitChildren(ir)
        return answer

    def visitForStmt(self, forstmt):
        id = forstmt.id
        answer = ForStmt(id = id)
        self._addVariable(id)
        answer.children = self._visitAll(forstmt.children)
        self._delVariable(id)
        return answer

    def visitSubscript(self, subscript):
        answer = Subscript()
        answer.children = self._visitAll(subscript.children)
        return answer

    def visitList(self, list):
        answer = List()
        answer.children = self._visitAll(list.children)
        return answer

    def visitCallStmt(self, call):
        answer = CallStmt(id = call.id)
        answer.children = self._visitAll(call.children)
        return answer

    def visitBlockStmt(self, block):
        answer = BlockStmt()
        answer.children = self._visitAll(block.children)
        return answer

    def visitSamplingStmt(self, sampling):
        target = sampling.target.accept(self)
        args = self._visitAll(sampling.args)
        return SamplingStmt(target = target, args = args,
                            id = sampling.id)

    def visitConditionalStmt(self, conditional):
        answer = ConditionalStmt()
        answer.children = self._visitAll(conditional.children)
        return answer
    

    def visitProgramBlock(self, block):
        self.block = block
        block.children = self._visitAll(block.children)
        return block

    def visitData(self, data):
        return self.visitProgramBlock(data)

    def visitModel(self, model):
        return self.visitProgramBlock(model)

    def visitParameters(self, params):
        return self.visitProgramBlock(params)

    def visitGuide(self, guide):
        return self.visitProgramBlock(guide)

    def visitPrior(self, prior):
        return self.visitProgramBlock(prior)
        
    def visitVariableDecl(self, decl):
        decl.dim = decl.dim.accept(self) if decl.dim else None
        name = decl.id
        self._addVariable(name)
        return decl

    def visitVariable(self, var):
        name = var.id
        if name not in self.ctx:
            assert False, "Use of undeclared variable:{name}".format(name)
        var.block = self.ctx[name].blockName()
        return var


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

    def loadAttr(self, obj, attr):
        return ast.Attribute(value = obj, attr = attr, ctx = ast.Load())

    def import_(self, name, asname = None):
        return ast.Import(names=[ast.alias(name=name, asname=asname)])

class Ir2PythonVisitor(IRVisitor):
    def __init__(self):
        super(Ir2PythonVisitor, self).__init__()
        self.data_names = set()
        self._in_prior = False
        self._in_guide = False
        self._guide_params = set()
        self._priors = {}
        self.target_name_visitor = TargetVisitor(self)
        self.helper = PythonASTHelper()

    def _ensureStmt(self, node):
        return self.helper.ensureStmt(node)

    def _ensureStmtList(self, listOrNode):
        return self.helper.ensureStmtList(listOrNode)

    def is_data(self, ir):
        target = ir.accept(self.target_name_visitor)
        return target in self.data_names

    def loadName(self, name):
        return self.helper.loadName(name)

    def import_(self, name, asname = None):
        return self.helper.import_(name, asname = asname)

    def loadAttr(self, obj, attr):
        return self.helper.loadAttr(obj, attr)

    def call(self, id,  args = [], keywords = []):
        return self.helper.call(id, args = args, keywords = keywords)

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
        return ast.Num(const.value)

    def visitVariableDecl(self, decl):
        if decl.data:
            self.data_names.add(decl.id)
        if self._in_guide:
            self._guide_params.add(decl.id)
        dims = decl.dim.accept(self) if decl.dim else None
        if decl.init:
            ## XXX
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
        assert var.block is not None
        return self.loadName(var.id)

    def visitCallStmt(self, call):
        return self._call(call.id, call.args)

    def visitAssignStmt(self, ir):
        target, value = self._visitAll((ir.target, ir.value))
        if ir.target.is_variable() and ir.target.is_guide_var():
            target_name = self.targetToName(target)
            call = self.call(self.loadAttr(self.loadName('pyro'), 'param'),
                            args = [
                                    target_name,
                                    value,
                                    ## XXX possible constraints
                            ])
            return self._assign(target, call)
        else:
            return self._assign(target, value)


    def _assign(self, target, value):
        if isinstance(target, ast.Name):
            return ast.Assign(
                    targets=[ast.Name(id = target.id, ctx = ast.Store())],
                    value = value)
        elif isinstance(target, ast.Subscript):
            target = ast.Subscript(value = target.value,
                                  slice = target.slice,
                                  ctx = ast.Store())
            return ast.Assign(
                    targets=[target],
                    value = value)
        assert False, "Don't know how to assign to {}".format(target)

    def _call(self, id_, args_):
        ## TODO id is a string!
        id = self.loadName(id_)
        args = args_.accept(self)
        args = args.elts if args else []
        return self.call(id, args=args)

    def visitForStmt(self, forstmt):
        ## TODO: id is not an object of ir!
        id = forstmt.id
        body, from_, to_ = self._visitAll((forstmt.body,
                                        forstmt.from_,
                                        forstmt.to_))
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
        left, right, op = self._visitAll((binaryop.left, 
                                        binaryop.right,
                                        binaryop.op))
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
        id, idx = self._visitAll((subscript.id, subscript.index))
        idx_z = ast.BinOp(left = idx, right = ast.Num(1), op = ast.Sub())
        return ast.Subscript(
                value = id,
                slice=ast.Index(value=idx_z),
                ctx=ast.Load())

    def visitSamplingStmt(self, sampling):
        target = sampling.target.accept(self)
        args = [arg.accept(self) for arg in sampling.args]
        id = sampling.id
        if hasattr(torch.distributions, id.capitalize()):
            # Check if the distribution exists in torch.distributions
            id = id.capitalize()
        ## if target is data, then we observe on that variable
        is_data = self.is_data(sampling.target)

        if is_data:
            keywords = [ast.keyword(arg='obs', value = target)]
        else:
            keywords = []
        
        dist = self.call(self.loadAttr(self.loadName('dist'), id),
                         args = args)
        if self._in_prior or (self._in_guide and \
                             not (sampling.target.is_variable() and sampling.target.is_params_var())):
            call = dist
        else:
            sample = self.loadAttr(self.loadName('pyro'), 'sample')
            target_name = self.targetToName(target)
            call = self.call(sample,
                            args = [target_name, dist],
                            keywords=keywords)
        if is_data:
            return ast.Expr(value = call)
        else:
            return self._assign(target, call)


    def visitNetVariable(self, var):
        prefix = 'prior' if self._in_prior else 'guide'
        name = self.loadName('{}_{}'.format(prefix, var.name))
        value = '.'.join(var.ids)  ##XXX
        return ast.Subscript(
                value = name,
                slice = ast.Index(value = ast.Str(value)),
                ctx = ast.Load()
        )


    def visitData(self, data):
        ## TODO: implement data behavior
        self._visitAll(data.body)
        return None

    def visitParameters(self, params):
        ## TODO: implement parameters behavior in here.
        return None

    def visitModel(self, model):
        body = self._visitAll(model.body)
        body = self._ensureStmtList(body)
        return self.buildModel(body)

    def visitPrior(self, prior):
        self._in_prior = True
        try:
            name = prior.body[0].target.name
            name_prior = 'prior_' + name ## XXX
            ## TODO: only one nn is suported in here.
            answer = self._visitAll(prior.body)
            body = [self._assign(self.loadName(name_prior), 
                                ast.Dict(keys = [], values=[])),]
            body += answer
            rand_mod = self.loadAttr(self.loadName('pyro'), 'random_module')
            lifted_name = 'lifted_' + name
            body += [
                self._assign(self.loadName(lifted_name),
                            self.call(rand_mod, 
                                        args = [ast.Str(name), 
                                                self.loadName(name),
                                                self.loadName(name_prior)])),
                ast.Return(value = self.call(self.loadName(lifted_name)))
            ]

                # lifted_module = pyro.random_module("mlp", mlp, priors)
                # return lifted_module()
            
            f = ast.FunctionDef(
                name = name_prior,
                args = ast.arguments(args = [],
                                    vararg = None,
                                    kwonlyargs = [],
                                    kw_defaults=[],
                                    kwarg=None,
                                    defaults=[]),
                body = body,
                decorator_list = [],
                returns = None
            )
        finally:
            self._in_prior = False
        self._priors = {name : name_prior}
        return f


    def visitGuide(self, guide):
        self._in_guide = True
        try:
            ## Hack!!! XXX
            ## a guide needs to know if it's working with a nn module
            ## as it should create the guide's dictionary and lift it
            body = guide.body
            is_net = isinstance(body[-1].target, NetVariable)
            name = guide.body[-1].target.name if is_net else  ''
            ## TODO: only one nn is suported in here.
            name_guide = 'guide_' + name ## XXX

            args = [ast.arg(name, None) for name in sorted(self.data_names)]
            inner_body = [x for x in self._visitAll(body) if x is not None]
            if is_net:
                body = [self._assign(self.loadName(name_guide), 
                                    ast.Dict(keys = [], values=[])),]
                body += inner_body
                rand_mod = self.loadAttr(self.loadName('pyro'), 'random_module')
                lifted_name = 'lifted_' + name
                body += [
                    self._assign(self.loadName(lifted_name),
                                self.call(rand_mod, 
                                            args = [ast.Str(name), 
                                                    self.loadName(name),
                                                    self.loadName(name_guide)])),
                    ast.Return(value = self.call(self.loadName(lifted_name)))
                ]
            else:
                body = inner_body
            
            f = ast.FunctionDef(
                name = name_guide,
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
        finally:
            self._in_guide = False
        return f

    def buildModel(self, body):
        args = [ast.arg(name, None) for name in sorted(self.data_names)]
        pre_body = []
        for prior in self._priors:
            pre_body.append(
                self._assign(self.loadName(prior),
                             self.call(self.loadName(self._priors[prior])))
            )
        model = ast.FunctionDef(
            name = 'model',
            args = ast.arguments(args = args,
                                 vararg = None,
                                 kwonlyargs = [],
                                 kw_defaults=[],
                                 kwarg=None,
                                 defaults=[]),
            body = pre_body + body,
            decorator_list = [],
            returns = None
        )
        return model

    def visitProgram(self, program):
        python_nodes = self._visitAll(program.blocks())
        body = self._ensureStmtList(python_nodes)
        module = ast.Module()
        module.body = [
            self.import_('torch'),
            self.import_('pyro'),
            self.import_('pyro.distributions', 'dist')]
        module.body += body
        ast.fix_missing_locations(module)
        astpretty.pprint(module)
        print(astor.to_source(module))
        return module

            


def ir2python(ir):
    annotator = VariableAnnotationsVisitor()
    ir = ir.accept(annotator)
    visitor = Ir2PythonVisitor()
    return ir.accept(visitor)



                                    