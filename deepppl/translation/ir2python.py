import ast
import torch
import astpretty
import astor

class IRVisitor(object):
    def visitProgram(self, ir):
        raise NotImplementedError

    def visitAssign(self, ir):
        raise NotImplementedError

    def visitSampling(self, ir):
        raise NotImplementedError

    def visitFor(self, ir):
        raise NotImplementedError

    def visitConditional(self, ir):
        raise NotImplementedError

    def visitWhile(self, ir):
        raise NotImplementedError

    def visitBlock(self, ir):
        raise NotImplementedError

    def visitCall(self, ir):
        raise NotImplementedError

    def visitConstant(self, ir):
        raise NotImplementedError

    def visitTuple(self, ir):
        raise NotImplementedError

    def visitStr(self, ir):
        raise NotImplementedError

    def visitBinary(self, ir):
        raise NotImplementedError

    def visitSubscript(self, ir):
        raise NotImplementedError

    def visitVariableDecl(self, ir):
        raise NotImplementedError

    def visitVariable(self, ir):
        raise NotImplementedError

class ISDataVisitor(IRVisitor):
    def __init__(self, ir2py):
        super(ISDataVisitor, self).__init__()
        self.ir2py = ir2py

    def checkContext(self, id):
        ctx = self.ir2py.context
        return id in ctx and ctx[id].data

    def visitVariable(self, var):
        return self.checkContext(var.id)

    def visitSubscript(self, subs):
        return subs.id.accept(self)

"Helper class for common `ast` objects"
class PythonASTHelper(object):
    def ensureStmt(self, node):
        if isinstance(node, ast.stmt):
            return node
        else:
            return ast.Expr(node)

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
        super(Ir2PythonVisitor).__init__()
        self.context = {}
        self.ast = []
        self.is_data_visitor = ISDataVisitor(self)
        self.helper = PythonASTHelper()

    def _ensureStmt(self, node):
        return self.helper.ensureStmt(node)

    def is_data(self, ir):
        return ir.accept(self.is_data_visitor)

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
        self.context[decl.id] = decl
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
        return ast.List(elts= [x.accept(self) for x in list.elements],
                        ctx = ast.Load())

    def visitVariable(self, var):
        return self.loadName(var.id)

    def visitCall(self, call):
        return self._call(call.id, call.args)

    def _call(self, id_, args_):
        ## TODO id is a string!
        id = self.loadName(id_)
        args = args_.accept(self)
        if args:
            args = [x for x in args.elts]
        else:
            args = []
        return self.call(id, args=args)

    def visitFor(self, forstmt):
        ## TODO: id is not an object of ir!
        id = forstmt.id
        body, from_, to_ = [x.accept(self) for x in (forstmt.body,
                                                    forstmt.from_,
                                                    forstmt.to_)]
        incl_to = ast.BinOp(left = to_, 
                            right = ast.Num(1), 
                            op = ast.Add())
        interval = [from_, incl_to]
        iter = self.call(self.loadName('range'), 
                                interval)
        body = self._ensureStmt(body)
        return ast.For(target = ast.Name(id = id, ctx=ast.Store()), 
                        iter = iter,
                        body = [body], ## XXX
                        orelse = [])       

    def visitSubscript(self, subscript):
        id, idx = [x.accept(self) for x in (subscript.id, subscript.index)]
        idx_z = ast.BinOp(left = idx, right = ast.Num(1), op = ast.Sub())
        return ast.Subscript(
                value = id,
                slice=ast.Index(value=idx_z),
                ctx=ast.Store())

    def visitSampling(self, sampling):
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
        target_name = self.targetToName(target)
        sample = self.loadAttr('pyro', 'sample')
        dist = self.call(self.loadAttr('dist', self.loadName(id)),
                         args = args)
        call = self.call(sample,
                        args = [target_name, dist],
                        keywords=keywords)
        if is_data:
            return call
        else:
            return ast.Assign(
                targets=[target],
                value = call)

    def visitProgram(self, ir):
        to_ast = lambda element: self._ensureStmt(element.accept(self))
        body = [to_ast(element) for element in ir.body]
        module = ast.Module()
        module.body = [
            self.import_('torch'),
            self.import_('pyro'),
            self.import_('pyro.distributions', 'dist')]
        module.body += [x for x in body if x]
        ast.fix_missing_locations(module)
        astpretty.pprint(module)
        print(astor.to_source(module))
            


def ir2python(ir):
    visitor = Ir2PythonVisitor()
    ir.accept(visitor)



                                    