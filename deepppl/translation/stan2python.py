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
import ast
from ast import *
from antlr4 import *
from parser.stanParser import stanParser
from parser.stanListener import stanListener
import astor
import astpretty


def parseExpr(expr):
    node = ast.parse(expr).body[0].value
    return node


def gatherChildrenAST(ctx):
    ast = []
    if ctx.children is not None:
        for child in ctx.children:
            if hasattr(child, 'ast') and child.ast is not None:
                ast += child.ast
        return ast


class Printer(stanListener):
    def __init__(self):
        self.indentation = 0

    def exitProgram(self, ctx):
        ctx.ast = Module()
        ctx.ast.body = [
            Import(names=[alias(name='torch', asname=None)]),
            ImportFrom(
                module='torch.distributions',
                names=[alias(name='*', asname=None)],
                level=0)]
        for child in ctx.children:
            if hasattr(child, 'ast'):
                ctx.ast.body += child.ast
        ast.fix_missing_locations(ctx.ast)
        # print(ast.dump(ctx.ast))
        print(astor.to_source(ctx.ast))

    def exitDataBlock(self, ctx):
        ctx.ast = gatherChildrenAST(ctx)

    def exitParametersBlock(self, ctx):
        ctx.ast = gatherChildrenAST(ctx)

    def exitModelBlock(self, ctx):
        ctx.ast = gatherChildrenAST(ctx)

    def exitExpressionCommaList(self, ctx):
        ctx.ast = gatherChildrenAST(ctx)

    def exitExpression(self, ctx):
        ctx.ast = [parseExpr(ctx.getText())]

    def exitStatementsOpt(self, ctx):
        ctx.ast = gatherChildrenAST(ctx)

    def exitStatement(self, ctx):
        if ctx.forStmt() is not None:
            ctx.ast = [ctx.forStmt().ast]
        if ctx.samplingStmt() is not None:
            ctx.ast = [ctx.samplingStmt().ast]

    def exitForStmt(self, ctx):
        id = ctx.IDENTIFIER().getText()
        body = []
        if hasattr(ctx.statement(), 'ast'):
            body = ctx.statement().ast
        if len(ctx.atom()) > 1:
            lbound = parseExpr(ctx.atom()[0].getText())
            ubound = parseExpr(ctx.atom()[1].getText())
            ctx.ast = For(
                target=Name(id=id, ctx=Store()),
                iter=Call(func=Name(
                    id='range', ctx=Load()),
                    args=[lbound, ubound],
                    keywords=[]),
                body=body,
                orelse=[])

    def exitSamplingStmt(self, ctx):
        lvalue = parseExpr(ctx.lvalueSampling().getText())
        if ctx.PLUS_EQ() is not None:
            assert False, 'Not yet implemented'
        else:
            id = ctx.IDENTIFIER()[0].getText().capitalize()
            exprList = ctx.expressionCommaList().ast
            ctx.ast = Assign(
                targets=[lvalue],
                value=Call(func=Attribute(
                    value=Call(func=Name(
                        id=id, ctx=Load()),
                        args=exprList,
                        keywords=[]),
                    attr='sample',
                    ctx=Load()),
                    args=[],
                    keywords=[]))

    def exitVariableDeclsOpt(self, ctx):
        ctx.ast = gatherChildrenAST(ctx)

    def exitVariableDecl(self, ctx):
        vid = ctx.IDENTIFIER().getText()
        if ctx.arrayDim() is not None:
            dims = parseExpr(ctx.arrayDim().getText())
            ctx.ast = [Assign(
                targets=[Name(id=vid, ctx=Store())],
                value=Call(func=Attribute(
                    value=Name(id='torch', ctx=Load()),
                    attr='zeros', ctx=Load()),
                    args=[dims],
                    keywords=[]))]
        else:
            ctx.ast = [Assign(
                targets=[Name(id=vid, ctx=Store())],
                value=Call(func=Attribute(
                    value=Name(id='torch', ctx=Load()),
                    attr='zeros', ctx=Load()),
                    args=[List(elts=[], ctx=Load())],
                    keywords=[]))]
