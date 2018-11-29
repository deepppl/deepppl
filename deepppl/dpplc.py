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
from antlr4.error.ErrorListener import ErrorListener
from .parser.stanLexer import stanLexer
from .parser.stanParser import stanParser
from .translation.stan2ir import StanToIR
from .translation.ir2python import ir2python

import ast
import astor
import torch
import pyro
import pyro.distributions as dist

class MyErrorListener(ErrorListener):
    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        print('Line ' + str(line) + ':' + str(column) +
              ': Syntax error, ' + str(msg))
        raise SyntaxError

def streamToParsetree(stream):
    lexer = stanLexer(stream)
    stream = CommonTokenStream(lexer)
    parser = stanParser(stream)
    parser._listeners = [MyErrorListener()]
    tree = parser.program()
    return tree

def parsetreeToIR(tree):
    toIr = StanToIR()
    walker = ParseTreeWalker()
    walker.walk(toIr, tree)
    return tree.ir

def stan2astpy(stream, verbose=False):
    tree = streamToParsetree(stream)
    ir = parsetreeToIR(tree)
    return ir2python(ir, verbose=verbose)

def stan2astpyFile(filename, verbose=False):
    stream = FileStream(filename)
    return stan2astpy(stream, verbose=verbose)

def stan2astpyStr(str, verbose=False):
    stream = InputStream(str)
    return stan2astpy(stream, verbose=verbose)

def stan2pystr(str, verbose=False):
    """Return the program's python source code""" 
    py = stan2astpyStr(str, verbose=verbose)
    return astor.to_source(py)

def do_compile(model_code = None, model_file = None, verbose=False):
    if not (model_code or model_file) or (model_code and model_file):
        assert False, "Either code or file but not both must be provided."
    if model_code:
        ast_ = stan2astpyStr(model_code, verbose=verbose)
    else:
        ast_ = stan2astpyFile(model_file, verbose=verbose)
    return compile(ast_, "<deepppl_ast>", 'exec')

def main(file, verbose=False):
    return stan2astpyFile(file, verbose=verbose)


if __name__ == '__main__':
    import argparse
    import pandas as pd
    parser = argparse.ArgumentParser(description='DeepPPL compiler')
    parser.add_argument('file', type=str,
                        help='A Stan file to compile')
    parser.add_argument('--print', action='store_true',
                        help='Print the generated Pyro code')
    parser.add_argument('--noinfer', action='store_true',
                    help='Do not launch inference')
    parser.add_argument('--verbose', action='store_true',
                    help='Output verbose code with shape information')
    args = parser.parse_args()
    verbose = False if args.verbose is None else args.verbose
    ast_ = main(args.file, verbose=verbose)
    if args.print:
        print(astor.to_source(ast_))
    co = compile(ast_, "<ast>", 'exec')
    eval(co)
    if not args.noinfer:
        x = torch.Tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
        posterior = pyro.infer.Importance(model, num_samples=1000)
        marginal = pyro.infer.EmpiricalMarginal(posterior.run(x), sites='theta')
        serie = pd.Series([marginal().item() for _ in range(1000)])
        print(serie.describe())
