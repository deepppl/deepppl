#%%
from deepppl.translation import ir2python
from deepppl  import dpplc
from antlr4 import *
from deepppl.parser.dppplLexer import dppplLexer
from deepppl.parser.dppplParser import dppplParser
from deepppl.translation.dpppl2ir import DpPPLToIR
from deepppl.translation import ir2python
from deepppl.translation import ir
import astor
import ipdb

print("hello")
import os
print(os.getcwd())


class VariableAnnotationsVisitorNew(ir2python.IRVisitor):
    def __init__(self):
        super(VariableAnnotationsVisitorNew, self).__init__()
        self.parameters = set()
        self._in_parameters = False
        self._in_guide = False 

    def defaultVisit(self, node):
        answer = node
        answer.children = self._visitChildren(node)
        return answer

    def visitVariableDecl(self, decl):
        if self._in_parameters:
            self.parameters.add(decl.id)
        return self.defaultVisit(decl)

    def visitGuide(self, guide):
        self._in_guide = True
        answer = self.defaultVisit(guide)
        self._in_guide = False
        return answer

    def visitVariable(self, var):
        if self._in_guide:
            import ipdb
            ipdb.set_trace()
        if self._in_guide and var.id in self.parameters:
            var.block_name = ir.GuideParameters.blockName()
        return self.defaultVisit(var)


    def visitGuideParameters(self, node):
        self._in_parameters = True
        answer = self.defaultVisit(node)
        self._in_parameters = False
        return answer

    def visitProgram(self, program):
        answer = ir.Program()
        answer.children = self._visitChildren(program)
        return answer


    
def _ir2python(ir):
    annotator = VariableAnnotationsVisitorNew()
    ir = ir.accept(annotator)
    #annotator = ir2python.VariableAnnotationsVisitor()
    #ir = ir.accept(annotator)
    nets = ir2python.NetworkVisitor()
    ir = ir.accept(nets)
    consistency = ir2python.SamplingConsistencyVisitor()
    ir = ir.accept(consistency)
    visitor = ir2python.Ir2PythonVisitor()
    return ir.accept(visitor)
#%%

#%%

#%%

filename = 'deepppl/tests/good/coin_guide.variation.stan'


def streamToParsetree(stream):
    lexer = dppplLexer(stream)
    stream = CommonTokenStream(lexer)
    parser = dppplParser(stream)
    tree = parser.program()
    return tree

def parsetreeToIR(tree):
    toIr = DpPPLToIR()
    walker = ParseTreeWalker()
    walker.walk(toIr, tree)
    return tree.ir

def file_to_ir(filename):
    stream = FileStream(filename)
    tree = streamToParsetree(stream)
    return _ir2python(parsetreeToIR(tree))

#%%
coin_ir = file_to_ir(filename)
print(astor.to_source(coin_ir))
