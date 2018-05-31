'''
 * Copyright 2018 IBM Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
'''

import sys
from antlr4 import *
from parser.stanParser import stanParser
from parser.stanListener import stanListener


class Printer(stanListener):
    def __init__(self):
        self.indentation = 0

    def dump(self, s):
        print(' ' * self.indentation * 4 + s)

    def enterProgram(self, ctx):
        self.dump('import torch')
        self.dump('from torch.distributions import *')

    def enterDataBlock(self, ctx):
        self.dump('\n# Data')

    def enterParametersBlock(self, ctx):
        self.dump('\n# Parameters')

    def enterModelBlock(self, ctx):
        self.dump('\n# Model')

    def enterForStmt(self, ctx):
        id = ctx.IDENTIFIER().getText()
        if len(ctx.atom()) > 1:
            lbound = ctx.atom()[0].getText()
            ubound = ctx.atom()[1].getText()
            self.dump('for ' + id + ' in range(' +
                      lbound + ',' + ubound + '):')
        else:
            it = ctx.atom()[0].getText()
            self.dump('for ' + id + ' in ' + it + ':')
        self.indentation += 1

    def exitForStmt(self, ctx):
        self.indentation -= 1

    def enterSamplingStmt(self, ctx):
        lvalue = ctx.lvalueSampling().getText()
        if ctx.PLUS_EQ() is not None:
            assert False, 'Not yet implemented'
        else:
            assert len(ctx.IDENTIFIER()) == 1
            id = ctx.IDENTIFIER()[0].getText()
            expr = ctx.expressionCommaList().getText()
            self.dump(lvalue + ' = ' + id.capitalize() +
                      '(' + expr + ').sample()')

    def enterVariableDecl(self, ctx):
        vid = ctx.IDENTIFIER().getText()
        if ctx.arrayDim() is not None:
            dims = ctx.arrayDim().getText()
            self.dump(vid + ' = torch.zeros(' + dims + ')')
        else:
            self.dump(vid + ' = torch.zeros([])')
