import ast
import torch
import astpretty
import astor

class IRVisitor(object):
    def visitProgram(self, ir):
        raise NotImplementedError

    def visitAssignStmt(self, ir):
        raise NotImplementedError

    def visitSamplingStmt(self, ir):
        raise NotImplementedError

    def visitForStmt(self, ir):
        raise NotImplementedError

    def visitConditionalStmt(self, ir):
        raise NotImplementedError

    def visitWhileStmt(self, ir):
        raise NotImplementedError

    def visitBlockStmt(self, ir):
        raise NotImplementedError

    def visitCallStmt(self, ir):
        raise NotImplementedError

    def visitConstant(self, ir):
        raise NotImplementedError

    def visitTuple(self, ir):
        raise NotImplementedError

    def visitStr(self, ir):
        raise NotImplementedError

    def visitBinaryOperator(self, ir):
        raise NotImplementedError

    def visitSubscript(self, ir):
        raise NotImplementedError

    def visitVariableDecl(self, ir):
        raise NotImplementedError

    def visitVariable(self, ir):
        raise NotImplementedError

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

    def _visitAll(self, iterable):
        filter = lambda x: (x is not None) or None
        return [filter(x) and x.accept(self) for x in iterable ]

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
        return self.loadName(var.id)

    def visitCallStmt(self, call):
        return self._call(call.id, call.args)

    def visitAssignStmt(self, ir):
        target, value = self._visitAll((ir.target, ir.value))
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
        if self._in_prior or (self._in_guide and sampling.target.id not in self._guide_params):
            call = dist
        elif self._in_guide:
            sample = self.call(self.loadAttr(dist, 'sample'))
            target_name = self.targetToName(target)
            call = self.call(self.loadAttr(self.loadName('pyro'), 'param'),
                            args = [
                                    target_name,
                                    sample,
                                    ## XXX possible constraints
                            ])
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
            name = guide.body[-1].target.name
            ## TODO: only one nn is suported in here.
            name_guide = 'guide_' + name ## XXX
            body = guide.body
            if body and type(body[0]) == type([]):
                self._visitAll(body[0])
                answer = self._visitAll(body[1:])
            else:
                answer = self._visitAll(body)
            args = [ast.arg(name, None) for name in sorted(self.data_names)]
            body = [self._assign(self.loadName(name_guide), 
                                ast.Dict(keys = [], values=[])),]
            body += answer
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

                # lifted_module = pyro.random_module("mlp", mlp, priors)
                # return lifted_module()
            
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
    visitor = Ir2PythonVisitor()
    return ir.accept(visitor)



                                    