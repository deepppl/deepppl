"""
 * Copyright 2018 
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
"""

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


class Config(object):
    def __init__(self, numpyro=False):
        self.numpyro = numpyro


class MyErrorListener(ErrorListener):
    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        print("Line " + str(line) + ":" + str(column) + ": Syntax error, " + str(msg))
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


def stan2astpy(stream, config, verbose=False):
    tree = streamToParsetree(stream)
    ir = parsetreeToIR(tree)
    return ir2python(ir, config, verbose=verbose)


def stan2astpyFile(filename, config, verbose=False):
    stream = FileStream(filename)
    return stan2astpy(stream, config, verbose=verbose)


def stan2astpyStr(str, config, verbose=False):
    stream = InputStream(str)
    return stan2astpy(stream, config, verbose=verbose)


def stan2pystr(str, config, verbose=False):
    """Return the program's python source code"""
    py = stan2astpyStr(str, config, verbose=verbose)
    return astor.to_source(py)


def do_compile(
    model_code=None, model_file=None, pyro_file=None, config=None, verbose=False
):
    if not any([model_code, model_file, pyro_file]):
        assert False, f"Either code, file or pyro file must be provided."
    if config is None:
        config = Config()
    if model_code:
        ast_ = stan2astpyStr(model_code, config, verbose=verbose)
    elif model_file:
        ast_ = stan2astpyFile(model_file, config, verbose=verbose)
    elif pyro_file:
        with open(pyro_file, "r") as file:
            ast_ = ast.parse(file.read())
    return compile(ast_, "<deepppl_ast>", "exec")


def main(file, verbose=False):
    config = Config()
    return stan2astpyFile(file, config, verbose=verbose)


if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="DeepPPL compiler")
    parser.add_argument("file", type=str, help="A Stan file to compile")
    parser.add_argument(
        "--print", action="store_true", help="Print the generated Pyro code"
    )
    parser.add_argument(
        "--noinfer", action="store_true", help="Do not launch inference"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Output verbose code with shape information",
    )
    args = parser.parse_args()
    verbose = False if args.verbose is None else args.verbose
    ast_ = main(args.file, verbose=verbose)
    if args.print:
        print(astor.to_source(ast_))
    co = compile(ast_, "<ast>", "exec")
    eval(co)
    if not args.noinfer:
        x = torch.Tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
        posterior = pyro.infer.Importance(model, num_samples=1000)
        marginal = pyro.infer.EmpiricalMarginal(posterior.run(x), sites="theta")
        serie = pd.Series([marginal().item() for _ in range(1000)])
        print(serie.describe())
