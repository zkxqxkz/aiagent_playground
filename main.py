"""
python main.py --problem 1
"""
import argparse
import importlib
import timeit
import sys

import numpy as np

from problems.api import Problem
from agent import QuestionAnsweringAgent
from sglang.lang.interpreter import ProgramState

MODEL_ID = "deepseek-ai/deepseek-coder-6.7b-instruct"

def run_agent(problem: Problem, ref_out):
    llvmir = problem.cfn_src

    print('Agent input:')
    print('\n================================')
    print(llvmir)
    print('================================\n')

    questions = [
        "Explain what the input code does",
        "Explain the main bottleneck in the input code is",
        "Suggest changes to the input IR and explain why they are supposed to help",
    ]
    
    # Initialize question answering agent
    qa_agent = QuestionAnsweringAgent(model=MODEL_ID)
    state: ProgramState = qa_agent(str(llvmir), questions)
    qa_agent.parse_output()
    qa_agent.shutdown()
    
    # Use the output from the above to optimize, based on the above
    
    """
    while i < max_retries:
        opt_agent = CodeOptimizationAgent(model=...)
        opt_agent()
        
        # validate output with the input
        optimized = opt_agent.get_optimized()
        
        problem.optimize(optimized)
        
        # if good save, 
        # if not good, check correctness, and use the error message to optimize
        # increament counter
        
    """
    
    
  

    breakpoint()
    # run your agent here!
    # TODO: for now just return a copy of the original IR
    optimized = str(llvmir)

    # try to compile the agent-generated IR
    problem.optimize(optimized)

    # after calling .optimize(), you can use "problem.ai_cfn(*ref_out)" to run your function
    # and perhaps compare it with the reference output
    # if you want to recompile, please call "problem.reset()" before calling "problem.optimize()"
    # again


def benchmark(fn, data):
    # return in milliseconds
    return timeit.timeit('fn(*data)', globals={ 'fn': fn, 'data': data }, number=100) * 1000


def check_the_same(a, b):
    assert a is not b
    if isinstance(a, np.ndarray):
        if a.dtype == np.float32:
            return np.allclose(a, b)
        else:
            return (a == b).all()
    return (a == b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', required=True, type=int, help='Problem number to run')
    args = parser.parse_args()

    pidx = args.problem
    try:
        pmod = importlib.import_module(f'problems.problem{pidx}')
    except:
        import traceback
        print(f'Failed to import the problem with the provided id={pidx}')
        traceback.print_exc()
        return 1

    p = pmod.problem

    ref = p.fn(*p.get_test_data())
    cref = p.cfn(*p.get_test_data())
    check_the_same(ref, cref)


    run_agent(p, ref)

    ai = p.ai_cfn(*p.get_test_data())
    if not check_the_same(cref, ai):
        raise ValueError('Output mismatch!')

    print('All outputs match. Benchmarking...')
    print('Base:', benchmark(p.fn, p.get_test_data()))
    print('Compiled:', benchmark(p.cfn, p.get_test_data()))
    print('AI-Opt:', benchmark(p.ai_cfn, p.get_test_data()))


if __name__ == '__main__':
    sys.exit(main() or 0)
