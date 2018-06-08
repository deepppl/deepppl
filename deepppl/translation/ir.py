class IR(object):
    def is_variable_decl(self):
        return False

    def accept(self, visitor):
        raise NotImplementedError
    
class Program(IR):
    def __init__(self, body = []):
        super(Program, self).__init__()
        self.body = body

    def accept(self, visitor):
        return visitor.visitProgram(self)

class ProgramBlocks(IR):
    pass

class Model(ProgramBlocks):
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
    
    def accept(self, visitor):
        return visitor.visitAssign(self)
    

class SamplingStmt(Statements):
    def __init__(self, target = None, id = None, args = None):
        super(SamplingStmt, self).__init__()
        self.target = target
        self.id = id
        self.args = args

    def accept(self, visitor):
        return visitor.visitSampling(self)   

class ForStmt(Statements):
    def __init__(self, id = None, from_ = None, to_ = None, body = None):
        super(ForStmt, self).__init__()
        self.id = id
        self.from_ = from_
        self.to_ = to_
        self.body = body

    def accept(self, visitor):
        return visitor.visitFor(self)   

class ConditionalStmt(Statements):
    def __init__(self, test = None, true = None, false = None):
        super(ConditionalStmt, self).__init__()
        self.test = test
        self.true = true
        self.false = false

    def accept(self, visitor):
        return visitor.visitConditional(self)   

class WhileStmt(Statements):
    def __init__(self, test = None, body = None):
        super(WhileStmt, self).__init__()
        self.test = test
        self.body = body

    def accept(self, visitor):
        return visitor.visitWhile(self)   

class BlockStmt(Statements):
    def __init__(self, body = None):
        super(BlockStmt, self).__init__()
        self.body = body

    def accept(self, visitor):
        return visitor.visitBlock(self)   

class CallStmt(Statements):
    def __init__(self, id = None, args = None):
        super(CallStmt, self).__init__()
        self.id = id
        self.args = args

    def accept(self, visitor):
        return visitor.visitCall(self)   

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

    def accept(self, visitor):
        return visitor.visitConstant(self)   

class Tuple(Expression):
    def __init__(self, exprs = None):
        super(Tuple, self).__init__()
        self.exprs = exprs

    def accept(self, visitor):
        return visitor.visitTuple(self)   

class Str(Expression):
    def __init__(self, value = None):
        super(Str, self).__init__()
        self.value = value

    def accept(self, visitor):
        return visitor.visitStr(self)

class List(Expression):
    def __init__(self, elements = []):
        super(List, self).__init__()
        self.elements = elements

    def accept(self, visitor):
        return visitor.visitList(self)  

class BinaryOperator(Expression):
    def __init__(self, left = None, op = None, right = None):
        super(BinaryOperator, self).__init__()
        self.left = left
        self.right = right
        self.op = op

    def accept(self, visitor):
        return visitor.visitBinary(self)   

class Subscript(Expression):
    def __init__(self, id = None, index = None):
        super(Subscript, self).__init__()
        self.id = id
        self.index = index

    def accept(self, visitor):
        return visitor.visitSubscript(self)   

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

    def accept(self, visitor):
        return visitor.visitVariableDecl(self)   

class Variable(Expression):
    def __init__(self, id = None):
        super(Variable, self).__init__()
        self.id = id

    def accept(self, visitor):
        return visitor.visitVariable(self)   

