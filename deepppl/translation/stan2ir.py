""" 
 Copyright 2018 IBM Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import sys
import ast
from parser.stanListener import stanListener
import astor
import astpretty
import torch
import ipdb
if __name__ is not None and "." in __name__:
    from .ir import *
    from .ir2python import *
else:
    assert False


def gatherChildrenIRList(ctx):
    irs = []
    if ctx.children is not None:
        for child in ctx.children:
            if hasattr(child, 'ir') and child.ir is not None:
                irs += child.ir
        return irs


def gatherChildrenIR(ctx):
    irs = []
    if ctx.children is not None:
        for child in ctx.children:
            if hasattr(child, 'ir') and child.ir is not None:
                irs.append(child.ir)
        return irs

def is_active(f):
    return f() is not None

def idxFromExprList(exprList):
    if len(exprList) == 1:
        return exprList[0]
    else:
        return Tuple(
            exprs = exprList)


class StanToIR(stanListener):
    def exitVariableDecl(self, ctx):
        vid = ctx.IDENTIFIER().getText()
        dims = ctx.arrayDim().ir if ctx.arrayDim() is not None else None
        ctx.ir = VariableDecl(id = vid, dim = dims)

    def exitArrayDim(self, ctx):
        ctx.ir = List(elements = ctx.expressionCommaList().ir)

    def exitVariableDeclsOpt(self, ctx):
        ctx.ir = gatherChildrenIR(ctx)

    # Vector, matrix and array expressions (section 4.2)


    def exitConstant(self, ctx):
        if ctx.IntegerLiteral() is not None:
            f = int
        elif ctx.RealLiteral() is not None:
            f = float
        else:
            assert False, "Unknonwn literal"
        ctx.ir = Constant(value = f(ctx.getText()))

    def exitVariable(self, ctx):
        ctx.ir = Variable(id = ctx.getText())

    def exitAtom(self, ctx):
        if is_active(ctx.constant):
            ctx.ir = ctx.constant().ir
        elif is_active(ctx.variable):
            ctx.ir = ctx.variable().ir
        elif is_active(ctx.expression):
            ctx.ir = ctx.expression().ir
        else:
            assert False, "Not yet implemented atom"

    def exitExpression(self, ctx):
        if is_active(ctx.atom):
            ctx.ir = ctx.atom().ir
            return
        if is_active(ctx.callExpr):
            ctx.ir = ctx.callExpr().ir
            return
        if ctx.TRANSPOSE_OP() is not None:
            assert False, "Not yet implemented"
        else:
            left = ctx.e1.ir
            right = ctx.e2.ir
            if is_active(ctx.LEFT_DIV_OP):
                assert False, "Not yet implemented"
            mapping = {
                ctx.PLUS_OP : Plus,
                ctx.MINUS_OP : Minus,
                ctx.POW_OP : Pow,
                ctx.OR_OP : Or,
                ctx.AND_OP : And,
                ctx.GT_OP : GT,
                ctx.LT_OP : LT,
                ctx.GE_OP : GE,
                ctx.LE_OP : LE,
                ctx.EQ_OP : EQ,
                ctx.DOT_DIV_OP : Div,
                ctx.DIV_OP : Div,
                ctx.DOT_MULT_OP : Mult,
                ctx.MULT_OP : Mult}
            op = None
            for src in mapping:
                if is_active(src):
                    op = mapping[src]()
                    break
            if op is not None:
                ctx.ir = BinaryOperator(left = left, 
                                        right = right, 
                                        op = op)
            elif '?' in ctx.getText():
                false = ctx.e3.ir
                ctx.ir = ConditionalStmt(test = left, 
                                        true = right,
                                        false = false)
            else:
                text = ctx.getText()
                assert False, "Not yet implemented: {}".format(text)

    def exitExpressionCommaList(self, ctx):
        ## TODO: check wheter we want to build a list of statements
        ## or a List node
        ctx.ir = gatherChildrenIR(ctx)

    def exitExpressionCommaListOpt(self, ctx):
        ctx.ir = gatherChildrenIRList(ctx)

    # Statements (section 5)

    # Assignment (section 5.1)

    def exitLvalue(self, ctx):
        id = Variable(ctx.IDENTIFIER().getText())
        if ctx.expressionCommaList() is not None:
            idx = idxFromExprList(ctx.expressionCommaList().ir)
            ctx.ir = Subscript(id = id, index = idx)
        else:
            ctx.ir = id

    def exitAssignStmt(self, ctx):
        lvalue = ctx.lvalue().ir
        expr = ctx.expression().ir
        if ctx.op is not None:
            op = None
            if ctx.PLUS_EQ() is not None:
                op = Plus()
            if ctx.MINUS_EQ() is not None:
                op = Minus()
            if ctx.MULT_EQ() or ctx.DOT_MULT_EQ() is not None:
                op = Mult()
            if ctx.DIV_EQ() or ctx.DOT_DIV_EQ() is not None:
                op = Div()
            if op:
                expr = BinaryOperator(left = lvalue, op = op, right = expr)
        ctx.ir = AssignStmt(
            target = lvalue,
            value = expr)

    # Sampling (section 5.3)

    def exitLvalueSampling(self, ctx):
        if is_active(ctx.lvalue):
            ctx.ir = ctx.lvalue().ir
        elif is_active(ctx.expression):
            ctx.ir = ctx.expression().ir
        elif is_active(ctx.netLValue):
            ctx.ir = ctx.netLValue().ir
        else:
            assert False


    def exitNetLValue(self, ctx):
        name = ctx.netName().getText()
        ids = [x.getText() for x in ctx.IDENTIFIER()]
        ctx.ir = NetVariable(name = name, ids = ids)

    def exitSamplingStmt(self, ctx):
        lvalue = ctx.lvalueSampling().ir
        if ctx.PLUS_EQ() is not None:
            assert False, 'Not yet implemented'
        elif ctx.truncation() is not None:
            assert False, 'Not yet implemented'
        else:
            id = ctx.IDENTIFIER()[0].getText()
            exprList = ctx.expressionCommaList().ir
            ctx.ir = SamplingStmt(target = lvalue,
                                  id = id,
                                  args = exprList)

    # For loops (section 5.4)

    def exitForStmt(self, ctx):
        id = ctx.IDENTIFIER().getText()
        body = ctx.statement().ir if hasattr(ctx.statement(), 'ir') else None
        atom = ctx.atom()
        from_ = atom[0].ir
        to_ = atom[1].ir if len(atom) > 1 else None
        ctx.ir = ForStmt(id = id, 
                        from_ = from_, 
                        to_ = to_, 
                        body = body)

    # Conditional statements (section 5.5)

    def exitConditionalStmt(self, ctx):
        test = ctx.expression().ir
        false = ctx.s2.ir if ctx.s2 is not None else None
        ctx.ir = ConditionalStmt(
            test = test,
            true = ctx.s1.ir,
            false = false,
        )

    # While loops (section 5.6)

    def exitWhileStmt(self, ctx):
        expr = ctx.expression().ir
        stmt = ctx.statement().ir
        ctx.ir = WhileStmt(
            test = expr,
            body = stmt
        )

    # Blocks (section 5.7)

    def exitBlockStmt(self, ctx):
        body = gatherChildrenIRList(ctx)
        ctx.ir = BlockStmt(body)

    # Functions calls (sections 5.9 and 5.10)

    def exitCallExpr(self, ctx):
        id = ctx.IDENTIFIER().getText()
        args = List(elements = ctx.expressionOrStringCommaList().ir)
        ctx.ir = CallStmt(id = id, args = args)

    def exitCallStmt(self, ctx):
        ctx.ir = ctx.callExpr().ir

    def exitExpressionOrString(self, ctx):
        if ctx.expression() is not None:
            ctx.ir = ctx.expression().ir
        else:
            ctx.ir = Str(value=ctx.getText())

    def exitExpressionOrStringCommaList(self, ctx):
        ctx.ir = gatherChildrenIR(ctx)

    # statements

    def exitStatement(self, ctx):
        if ctx.assignStmt() is not None:
            ctx.ir = ctx.assignStmt().ir
        if ctx.samplingStmt() is not None:
            ctx.ir = ctx.samplingStmt().ir
        if ctx.forStmt() is not None:
            ctx.ir = ctx.forStmt().ir
        if ctx.conditionalStmt() is not None:
            ctx.ir = ctx.conditionalStmt().ir
        if ctx.whileStmt() is not None:
            ctx.ir = ctx.whileStmt().ir
        if ctx.blockStmt() is not None:
            ctx.ir = ctx.blockStmt().ir
        if ctx.callStmt() is not None:
            ctx.ir = ctx.callStmt().ir
        if ctx.BREAK() is not None:
            ctx.ir = BreakStmt()
        if ctx.CONTINUE() is not None:
            ctx.ir = ContinueStmt()

    def exitStatementsOpt(self, ctx):
        ctx.ir = gatherChildrenIR(ctx)

    # Program blocks (section 6)

    def exitDataBlock(self, ctx):
        body = gatherChildrenIRList(ctx)
        for ir in body:
            if ir.is_variable_decl():
                ir.set_data()
        ctx.ir = Data(body = body)

    def exitParametersBlock(self, ctx):
        body = gatherChildrenIRList(ctx)
        ctx.ir = Parameters(body = body)

    def exitGuideBlock(self, ctx):
        body = gatherChildrenIR(ctx)
        ctx.ir = Guide(body = body)

    def exitPriorBlock(self, ctx):
        # TODO: unify gatherChildrenIRList: check for StatemetnsOpt
        body = gatherChildrenIR(ctx)
        ctx.ir = Prior(body = body)

    def exitModelBlock(self, ctx):
        body = gatherChildrenIRList(ctx)
        ctx.ir = Model(body= body)

    def exitProgram(self, ctx):
        body = []
        for child in ctx.children:
            if hasattr(child, 'ir') and child.ir is not None:
                body.append(child.ir)
        ctx.ir = Program(body = body)

