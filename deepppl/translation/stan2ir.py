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
from ..parser.stanListener import stanListener
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
    return f is not None and f() is not None

def idxFromExprList(exprList):
    if len(exprList) == 1:
        return exprList[0]
    else:
        return Tuple(
            exprs = exprList)


class StanToIR(stanListener):
    def __init__(self):
        self.networks = None
        self._to_model = []

    def exitVariableDecl(self, ctx):
        vid = ctx.IDENTIFIER().getText()
        type_ = ctx.type_().ir
        dims = None
        if ctx.arrayDim() is not None and type_.dim is not None:
            # Avi: to check
            dims = Tuple(exprs = [ctx.arrayDim().ir, type_.dim])
        elif ctx.arrayDim() is not None:
            dims = ctx.arrayDim().ir
        elif type_.dim is not None:
            dims = type_.dim
        init = ctx.expression().ir if is_active(ctx.expression) else None
        ctx.ir = VariableDecl(
                    id = vid,
                    dim = dims,
                    type_ = type_,
                    init = init)

    def exitType_(self, ctx):
        ptype = ctx.primitiveType()
        if ctx.primitiveType() is not None:
            type_ = ctx.primitiveType().getText()
        elif ctx.vectorType() is not None:
            type_ = ctx.vectorType().getText()
        elif ctx.matrixType() is not None:
            type_ = ctx.matrixType().getText()
        else:
            assert False, f"unknown type: {ptype.getText()}"
        constraints = ctx.typeConstraints().ir if ctx.typeConstraints() else None
        is_array = ctx.isArray is not None
        dims = ctx.arrayDim().ir if ctx.arrayDim() is not None else None
        ctx.ir = Type_(type_ = type_, constraints = constraints, is_array = is_array, dim = dims)

    def exitTypeConstraints(self, ctx):
        constraints_list = ctx.typeConstraintList()
        if constraints_list:
            ctx.ir = [x.ir for x in constraints_list.typeConstraint()]

    def exitTypeConstraint(self, ctx):
        id_ = ctx.IDENTIFIER()
        if id_.getText() == 'lower':
            sort = 'lower'
        elif id_.getText() == 'upper':
            sort = 'upper'
        else:
            assert False, f'unknown constraint: {id_.getText()}'
        constant = ctx.atom().ir
        constraint = Constraint(sort = sort, value = constant)
        ctx.ir = constraint

    def exitArrayDim(self, ctx):
        cl = ctx.expressionCommaList()
        if cl:
            elements = cl.ir
            if len(elements) == 1:
                ctx.ir = elements[0]
            else:
                ctx.ir = Tuple(exprs = elements)
        elif ctx.inferredArrayShape():
            ctx.ir = AnonymousShapeProperty()

    def exitParameterDecl(self, ctx):
        if is_active(ctx.variableDecl):
            ctx.ir = ctx.variableDecl().ir
        else: # Could be more defensive
            pass

    def exitParameterDeclsOpt(self, ctx):
        ctx.ir = gatherChildrenIR(ctx)

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

    def exitIndexExpression(self, ctx):
        if is_active(ctx.expressionCommaListOpt):
            ctx.ir = ctx.expressionCommaListOpt().ir
        else:
            assert False, "Unknown index expression:{}.".format(ctx.getText())

    def exitAtom(self, ctx):
        if is_active(ctx.constant):
            ctx.ir = ctx.constant().ir
        elif is_active(ctx.variable):
            ctx.ir = ctx.variable().ir
        elif is_active(ctx.expression):
            ctx.ir = ctx.expression().ir
        elif is_active(ctx.atom) and is_active(ctx.indexExpression):
            name = ctx.atom().ir
            index = ctx.indexExpression().ir
            ctx.ir = Subscript(id = name, index = index)
        elif is_active(ctx.netLValue):
            ctx.ir = ctx.netLValue().ir
        elif is_active(ctx.variableProperty):
            ctx.ir = ctx.variableProperty().ir
        else:
            assert False, "Not yet implemented atom: {}".format(ctx.getText())

    def exitExpression(self, ctx):
        if is_active(ctx.atom):
            ctx.ir = ctx.atom().ir
            return
        if is_active(ctx.callExpr):
            ctx.ir = ctx.callExpr().ir
            return
        if ctx.TRANSPOSE_OP() is not None:
            assert False, "Not yet implemented"
        elif ctx.e1 is not None and ctx.e2 is not None:
            self._exitBinaryExpression(ctx)
        elif ctx.e1 is not None:
            if is_active(ctx.PLUS_OP):
                op = UPlus()
            elif is_active(ctx.MINUS_OP):
                op = UMinus()
            elif is_active(ctx.NOT_OP):
                op = UNot()
            else:
                assert False, f'Unknown operator: {ctx.getText()}'
            ctx.ir = UnaryOperator(
                        value = ctx.e1.ir,
                        op = op)
        else:
            text = ctx.getText()
            assert False, "Not yet implemented: {}".format(text)

    def _exitBinaryExpression(self, ctx):
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
        elif ctx.e3 is not None:
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
        ir = gatherChildrenIRList(ctx)
        if len(ir) == 1:
            ctx.ir = ir[0]
        else:
            ctx.ir = Tuple(exprs = ir)

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

    def exitNetParam(self, ctx):
        ids = [ctx.IDENTIFIER().getText()]
        if ctx.netParam():
            ir = ctx.netParam()[0].ir
            ids.extend(ir)
        ctx.ir = ids

    def exitNetworksBlock(self, ctx):
        ops = ctx.netVariableDeclsOpt()
        decls = [x.ir for x in ops.netVariableDecl()]
        nets = NetworksBlock(decls = decls)
        self.networks = nets
        ctx.ir = nets

    def exitNetClass(self, ctx):
        ctx.ir = ctx.getText()

    exitNetName = exitNetClass

    def exitNetVariableDecl(self, ctx):
        netCls = ctx.netClass().ir
        name = ctx.netName().ir
        parameters = []
        ctx.ir = NetDeclaration(name = name, cls = netCls, \
                                params = parameters)

    def exitNetParamDecl(self, ctx):
        netName = ctx.netName().getText()
        if self.networks is not None:
            nets = [x for x in self.networks.decls if x.name == netName]
            if len(nets) == 1:
                nets[0].params.append(ctx.netParam().ir)
            elif len(nets) > 1:
                raise AlreadyDeclaredException(netName)
            else:
                raise UndeclaredNetworkException(netName)
        else:
            raise UndeclaredNetworkException(netName)

    def exitNetLValue(self, ctx):
        name = ctx.netName().getText()
        ids = ctx.netParam().ir
        ctx.ir = NetVariable(name = name, ids = ids)

    def exitVariableProperty(self, ctx):
        property = ctx.IDENTIFIER().getText()
        if is_active(ctx.netLValue):
            var = ctx.netLValue().ir
            cls = NetVariableProperty
        elif is_active(ctx.variable):
            var = ctx.variable().ir
            cls = VariableProperty
        else:
            assert False, "Not yet implemented."
        ctx.ir = cls(var = var, prop= property)

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
        args = ctx.expressionOrStringCommaList().ir
        ctx.ir = CallStmt(id = id, args = args)

    def exitCallStmt(self, ctx):
        ctx.ir = ctx.callExpr().ir

    def exitExpressionOrString(self, ctx):
        if ctx.expression() is not None:
            ctx.ir = ctx.expression().ir
        else:
            ctx.ir = Str(value=ctx.getText())

    def exitExpressionOrStringCommaList(self, ctx):
        ir = gatherChildrenIR(ctx)
        elements = ir if ir is not None else []
        ctx.ir = List(elements = elements)

    # statements

    def exitStatement(self, ctx):
        if ctx.assignStmt() is not None:
            ctx.ir = ctx.assignStmt().ir
        if ctx.samplingStmt() is not None:
            ctx.ir = ctx.samplingStmt().ir
        if ctx.incrementLogProbStmt() is not None:
            ctx.ir = ctx.incrementLogProbStmt().ir
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

    def exitIncrementLogProbStmt(self, ctx):
        ctx.ir = SamplingFactor(target=ctx.expression().ir)

    def exitStatementsOpt(self, ctx):
        ctx.ir = gatherChildrenIR(ctx)

    # Program blocks (section 6)

    def exitDataBlock(self, ctx):
        body = gatherChildrenIRList(ctx)
        for ir in body:
            if ir.is_variable_decl():
                ir.set_data()
        ctx.ir = Data(body = body)

    def exitTransformedDataBlock(self, ctx):
        body = gatherChildrenIRList(ctx)
        for ir in body:
            if ir.is_variable_decl():
                ir.set_transformed_data()
        ctx.ir = TransformedData(body = body)

    def code_block(self, ctx, cls):
        body = gatherChildrenIRList(ctx)
        ctx.ir = cls(body = body)

    def exitParametersBlock(self, ctx):
        self.code_block(ctx, Parameters)
        for ir in ctx.ir.body:
            if ir.is_variable_decl():
                ir.set_parameters()

    def exitTransformedParametersBlock(self, ctx):
        body = gatherChildrenIRList(ctx)
        for ir in body:
            if ir.is_variable_decl():
                ir.set_transformed_parameters()
        self._to_model = body
        ctx.ir = TransformedParameters(body = body)

    def exitGuideBlock(self, ctx):
        self.code_block(ctx, Guide)

    def exitGuideParametersBlock(self, ctx):
        self.code_block(ctx, GuideParameters)

    def exitPriorBlock(self, ctx):
        # TODO: unify gatherChildrenIRList: check for StatemetnsOpt
        body = gatherChildrenIR(ctx)
        ctx.ir = Prior(body = body)

    def exitModelBlock(self, ctx):
        body = gatherChildrenIRList(ctx)
        ctx.ir = Model(body= self._to_model + body)

    def exitGeneratedQuantitiesBlock(self, ctx):
        body = gatherChildrenIRList(ctx)
        for ir in body:
            if ir.is_variable_decl():
                ir.set_generated_quatities()
        ctx.ir = GeneratedQuantities(body= body)

    def exitProgram(self, ctx):
        body = []
        for child in ctx.children:
            if hasattr(child, 'ir') and child.ir is not None:
                body.append(child.ir)
        ctx.ir = Program(body = body)

