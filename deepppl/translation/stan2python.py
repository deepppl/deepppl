import sys
from antlr4 import *
from parser.stanParser import stanParser
from parser.stanListener import stanListener


class Printer(stanListener):
    def exitSamplingStmt(self, ctx):
        print('--------------')
        print(ctx.getText())
        print(dir(ctx))
        print(ctx.getTypedRuleContext(stanParser.LvalueContext, 1))
        print(ctx.getTypedRuleContexts(stanParser.SamplingStmtContext))
    def exitDataBlock(self, ctx):
        print("Oh, a data!")
    def exitParametersBlock(self, ctx):
        print("Oh, a parameters!")
    def exitModelBlock(self, ctx):
        print("Oh, a model!")
