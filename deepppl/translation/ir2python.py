import ast
import torch

class IRVisitor(object):
    def visitProgam(self, ir):
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

class Ir2PythonVisitor(IRVisitor):
    def __init__(self):
        super(Ir2PythonVisitor).__init__()
        self.context = {}
        self.ast = []
        self.is_data_visitor = ISDataVisitor(self)


    def visitConstant(self, const):
        return ast.Num(const.value)

    def visitVariableDecl(self, decl):
        self.context[decl.id] = decl
        dims = decl.dim.accept(self) if decl.dim else None
        if decl.init:
            assert not decl.data, "Data cannot be initialized"
            init = decl.init.accept(self)
            return ast.Assign(
                    targets=[ast.Name(id=decl.id, ctx=ast.Store())],
                    value=ast.Call(func=ast.Attribute(
                        value=ast.Name(id='torch', ctx=ast.Load()),
                        attr='zeros', ctx=ast.Load()),
                        args=[ast.List(elts=dims, ctx=ast.Load())],
                        keywords=[]))
        return None

    def is_data(self, ir):
        return ir.accept(self.is_data_visitor)

    def visitList(self, list):
        return ast.List(elts= [x.accept(self) for x in list.elements],
                        ctx = ast.Load())

    def visitVariable(self, var):
        return ast.Name(id = var.id, ctx = ast.Load())

    def visitCall(self, call):
        ## TODO id is a string!
        id = ast.Name(id = call.id, ctx=ast.Load())
        args = call.args.accept(self)
        if args:
            args = [x for x in args.elts]
        else:
            args = []
        return  ast.Call(func=id,
                        args=args,
                        keywords=[])

    def visitFor(self, forstmt):
        ## TODO: id is not an object of ir!
        id = forstmt.id
        from_, to_, body = [x.accept(self) for x in \
                                    (forstmt.from_, forstmt.to_,
                                    forstmt.body)]
        return ast.For(target = ast.Name(id = id, ctx=ast.Store()), 
                        iter = ast.Call(func = ast.Name(id = 'range', 
                                                        ctx= ast.Load()),
                                        args = [from_, to_],
                                        keywords=[]),
                        body = [ast.Expr(body)], ## XXX
                        orelse = [])       

    def visitSubscript(self, subscript):
        id, idx = [x.accept(self) for x in (subscript.id, subscript.index)]
        ## TODO: transform index to zero-based
        return ast.Subscript(
                value = id,
                slice=ast.Index(value=idx),
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
        call = ast.Call(func= ast.Attribute(
                    value=ast.Call(
                        func= ast.Name(id=id, ctx=ast.Load()),
                        args=args,
                        keywords=[]),
                    attr='sample',
                    ctx=ast.Load()),
                    args=[],
                    keywords=keywords)
        if is_data:
            return call
        else:
            return ast.Assign(
                targets=[target],
                value = call)
            



                                    