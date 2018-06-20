class IRMeta(type):
    def __init__(cls, *args, **kwargs):
        selector = 'return visitor.visit{}(self)'.format(cls.__name__)
        accept_code = "def accept(self, visitor):\n\t{}".format(selector)
        l = {}
        exec(accept_code, globals(), l)
        setattr(cls, "accept", l["accept"])

class IR(metaclass=IRMeta):
    def is_variable_decl(self):
        return False

    def is_variable(self):
        return False

    @property
    def children(self):
        return []

    @children.setter
    def children(self, children):
        assert not children, "Setting children to object without children"
    
class Program(IR):
    def __init__(self, body = []):
        super(Program, self).__init__()
        self.body = body
        self.addBlocks(body)

    @property
    def children(self):
        return self.blocks()

    @children.setter
    def children(self, blocks):
        self.addBlocks(blocks)

    def addBlocks(self, blocks):
        for block in blocks:
            name = block.blockName()
            if getattr(self, name, None) is not None:
                assert False, "trying to set :{} twice.".format(name)
            setattr(self, name, block)
    
    def blocks(self):
        ## impose an evaluation order
        blockNames = ['data', 'parameters', 'guide', 'prior', 'model']
        for name in blockNames:
            block = getattr(self, name, None)
            if block is not None:
                yield block


        

class ProgramBlocks(IR):
    def __init__(self, body = []):
        super(ProgramBlocks, self).__init__()
        self.body = body

    @property
    def children(self):
        return self.body

    @children.setter
    def children(self, body):
        self.body = body

    @classmethod
    def blockName(cls):
        return cls.__name__.lower()

class Model(ProgramBlocks):
    pass

class Guide(ProgramBlocks):
    pass

class Prior(ProgramBlocks):
    pass

class Parameters(ProgramBlocks):
    pass

class Data(ProgramBlocks):
    pass

class Statements(IR):
    pass

class AssignStmt(Statements):
    def __init__(self, target = None, value = None):
        super(AssignStmt, self).__init__()
        self.target = target
        self.value = value

    @property
    def children(self):
        return [self.target, self.value]

    @children.setter
    def children(self, children):
        [self.target, self.value] = children

class SamplingStmt(Statements):
    def __init__(self, target = None, id = None, args = None):
        super(SamplingStmt, self).__init__()
        self.target = target
        self.id = id
        self.args = args

    @property
    def children(self):
        return [self.target, self.args]

    @children.setter
    def children(self, children):
        [self.target, self.args] = children

class ForStmt(Statements):
    def __init__(self, id = None, from_ = None, to_ = None, body = None):
        super(ForStmt, self).__init__()
        self.id = id
        self.from_ = from_
        self.to_ = to_
        self.body = body


    @property
    def children(self):
        return [self.from_, self.to_, self.body]

    @children.setter
    def children(self, children):
        [self.from_, self.to_, self.body] = children

class ConditionalStmt(Statements):
    def __init__(self, test = None, true = None, false = None):
        super(ConditionalStmt, self).__init__()
        self.test = test
        self.true = true
        self.false = false

    @property
    def children(self):
        return [self.test, self.true, self.false]

    @children.setter
    def children(self, children):
        [self.test, self.true, self.false] = children

class WhileStmt(Statements):
    def __init__(self, test = None, body = None):
        super(WhileStmt, self).__init__()
        self.test = test
        self.body = body

    @property
    def children(self):
        return [self.test, self.body]

    @children.setter
    def children(self, children):
        [self.test, self.body] = children

class BlockStmt(Statements):
    def __init__(self, body = None):
        super(BlockStmt, self).__init__()
        self.body = body

    @property
    def children(self):
        return self.body

    @children.setter
    def children(self, children):
        self.body = children

class CallStmt(Statements):
    def __init__(self, id = None, args = None):
        super(CallStmt, self).__init__()
        self.id = id
        self.args = args

    @property
    def children(self):
        return [self.args]

    @children.setter
    def children(self, children):
        [self.args,] = children  


class BreakStmt(Statements):
    pass

class ContinueStmt(Statements):
    pass

class Expression(Statements):
    pass

class Constant(Expression):
    def __init__(self, value = None):
        super(Constant, self).__init__()
        self.value = value


class Tuple(Expression):
    def __init__(self, exprs = None):
        super(Tuple, self).__init__()
        self.exprs = exprs

    @property
    def children(self):
        return self.exprs

    @children.setter
    def children(self, children):
        self.exprs = children

class Str(Expression):
    def __init__(self, value = None):
        super(Str, self).__init__()
        self.value = value

class List(Expression):
    def __init__(self, elements = []):
        super(List, self).__init__()
        self.elements = elements

    @property
    def children(self):
        return self.elements

    @children.setter
    def children(self, children):
        self.elements = children

class BinaryOperator(Expression):
    def __init__(self, left = None, op = None, right = None):
        super(BinaryOperator, self).__init__()
        self.left = left
        self.right = right
        self.op = op 

    @property
    def children(self):
        return [self.left, self.right, self.op]

    @children.setter
    def children(self, children):
        [self.left, self.right, self.op] = children

class Subscript(Expression):
    def __init__(self, id = None, index = None):
        super(Subscript, self).__init__()
        self.id = id
        self.index = index 

    @property
    def children(self):
        return [self.id, self.index]

    @children.setter
    def children(self, children):
        [self.id, self.index] = children

class VariableDecl(IR):
    def __init__(self, id = None, dim = None, init = None):
        super(VariableDecl, self).__init__()
        self.id = id
        self.dim = dim
        self.init = init
        self.data = False

    def is_variable_decl(self):
        return True

    def set_data(self):
        self.data = True

    @property
    def children(self):
        return [self.dim, self.init]

    @children.setter
    def children(self, children):
        [self.dim, self.init] = children

class Variable(Expression):
    def __init__(self, id = None):
        super(Variable, self).__init__()
        self.id = id
        self.block = None

    def is_variable(self):
        return True

    def is_data_var(self):
        return self.block == Data.blockName()

    def is_params_var(self):
        return self.block == Parameters.blockName()

    def is_guide_var(self):
        return self.block == Guide.blockName()

    def is_prior_var(self):
        return self.block == Prior.blockName()

class NetVariable(Expression):
    def __init__(self, name = None, ids =  []):
        super(NetVariable, self).__init__()
        self.name = name
        self.ids = ids

    def getid(self):
        return self.name + '.'.join(self.ids)

    id = property(getid)

class Operator(Expression):
    pass

class Plus(Operator):
    pass 

class Minus(Operator):
    pass 

class Pow(Operator):
    pass

class Mult(Operator):
    pass  

class Div(Operator):
    pass

class And(Operator):
    pass

class Or(Operator):
    pass

class LE(Operator):
    pass

class GE(Operator):
    pass

class LT(Operator):
    pass

class GT(Operator):
    pass

class EQ(Operator):
    pass


