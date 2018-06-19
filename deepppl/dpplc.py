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
from parser.stanLexer import stanLexer
from parser.stanParser import stanParser
from translation.stan2ir import StanToIR
from translation.ir2python import ir2python


import torch
import pyro
import pyro.distributions as dist

def streamToParsetree(stream):
    lexer = stanLexer(stream)
    stream = CommonTokenStream(lexer)
    parser = stanParser(stream)
    tree = parser.program()
    return tree

def parsetreeToIR(tree):
    toIr = StanToIR()
    walker = ParseTreeWalker()
    walker.walk(toIr, tree)
    return tree.ir

def stan2astpy(stream):
    tree = streamToParsetree(stream)
    ir = parsetreeToIR(tree)
    return ir2python(ir)

def stan2astpyFile(filename):
    stream = FileStream(filename)
    return stan2astpy(stream)

def main(argv):
    return stan2astpyFile(argv[1])


if __name__ == '__main__':
    import pandas as pd
    ast_ = main(sys.argv)
    co = compile(ast_, "<ast>", 'exec')
    eval(co)
    x = torch.Tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    posterior = pyro.infer.Importance(model, num_samples=1000)
    marginal = pyro.infer.EmpiricalMarginal(posterior.run(x), sites='theta')
    serie = pd.Series([marginal().item() for _ in range(1000)])
    print(serie.describe())
