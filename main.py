"""
python main.py --problem 1
"""
import argparse
import importlib
import timeit
import sys
import time
import numpy as np

from problems.api import Problem
import inspect

from agent.deepseek_agent import OptimizationPlanningAgent
from sglang.lang.interpreter import ProgramState


from agent.openai_agent import OAIAgent

MODEL_ID = "deepseek-ai/deepseek-coder-6.7b-instruct"

def run_agent(problem: Problem, ref_out, max_retries: int = 3):
    llvmir = problem.cfn_src

    print('Agent input:')
    print('\n================================')
    print(llvmir)
    print('================================\n')


    """
    # This method uses sglang and deepseek 
    
    # Using Open source model (deepseek) tends to hallucinate, will need to finetune.
    # Afterwards, consider using trl PPOTrainer and PPOConfig to provide feedback
    # ex. based on correctness (0 or 1) and the run time (1.0 - (optimized_time / original_time))
    # then final_reward = correctness_reward + performance_reward
    # use the reward for ppo_trainer.step(..., reward)
    
    # Use the output from the above to optimize, based on the above
    llvmir_optimization_agent = OptimizationPlanningAgent(model=MODEL_ID)
    llvmir_optimization_agent(
        str(llvmir), 
        inspect.getsource(problem.fn),
        problem.test_data,
        problem.cfn(*problem.get_test_data()),
        benchmark(problem.fn, problem.get_test_data()),
        problem.cpu_info,
    ) # infernce
    llvmir_optimization_agent.parse_output()
    plan = llvmir_optimization_agent.genrate_plan()
    print(plan)
    breakpoint()
    llvmir_optimization_agent.shutdown()
    
    """

    cref = problem.cfn(*problem.get_test_data())
    
    agent = OAIAgent(model="gpt-4o")
    current_llvmir = str(llvmir)

    output = problem.cfn(*problem.get_test_data())
    runtime = benchmark(problem.fn, problem.get_test_data())
    failed_generation = failed_plan = error = None
    i = 0

    while i < max_retries:
        print("Iter: ", i)
        problem.reset()
        plan_file_name = f"./plan_{i}_{problem.fn.__name__}.txt"
        
        agent.create_plan(
            llvm_code=current_llvmir, 
            python_code=inspect.getsource(problem.fn),
            test_data=problem.test_data,
            output=problem.cfn(*problem.get_test_data()),
            runtime=benchmark(problem.fn, problem.get_test_data()),
            cpu_info=problem.cpu_info,
            failed_plan = failed_plan,
            failed_generation=failed_generation,
            error=error,
            save_path=plan_file_name,
            cfn_name=problem._cfn_name,
        )
        optimized = agent.optimize(
            path=plan_file_name,
            llvm_code=current_llvmir, 
            cfn_name=problem._cfn_name,
            test_data=problem.test_data,
            output=output,
            runtime=runtime,
        )

        print('\n================================')
        print('Agent output:\n')
        print(optimized)
        print('================================\n')

        # save the optimized 
        plan_file_name = f"./out_{i}_{problem.fn.__name__}.txt"
        with open(plan_file_name, "w") as f:
            f.write(optimized)
        try:
            problem.optimize(optimized)
            ai = problem.ai_cfn(*problem.get_test_data())
            
            if check_the_same(cref, ai):
                
                # benchmark
                total_time = 0.0
                N = 1000
                for _ in range(N):
                    total_time += benchmark(problem.ai_cfn, problem.get_test_data())
                
                runtime_avg = total_time / N
                print("average run-time: ", runtime_avg)
                current_llvmir=optimized
                agent.optimized_codes.append((runtime_avg, optimized))
                failed_generation = failed_plan = error = None
                
            else:
                # provide failed generation that produces different output
                failed_generation = optimized
                with open(plan_file_name, "r", encoding="utf-8") as file:
                    failed_plan = file.read()
                error = None
        except Exception as err:
            print(f"Could not process\n{optimized}\n\n, got error {err}")
            failed_generation = optimized
            error = err
            failed_plan = None
        i += 1
        
    problem.reset()
    
    # get the fastest code
    agent.optimized_codes.sort(key=lambda x: x[0])
    best_ai_output = agent.optimized_codes[0][1]
    problem.optimize(best_ai_output)


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

    run_agent(p, ref, max_retries=5)   # get the best of the three tries

    ai = p.ai_cfn(*p.get_test_data())
    if not check_the_same(cref, ai):
        raise ValueError('Output mismatch!')

    print('All outputs match. Benchmarking...')
    print('Base:', benchmark(p.fn, p.get_test_data()))
    print('Compiled:', benchmark(p.cfn, p.get_test_data()))
    print('AI-Opt:', benchmark(p.ai_cfn, p.get_test_data()))


if __name__ == '__main__':
    sys.exit(main() or 0)
