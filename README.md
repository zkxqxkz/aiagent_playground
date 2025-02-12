

# AI agent playground

This repo provides a simple playground for an AI agent to work on low-level optimizations.
It uses numba and LLVM, as well as 3 example simple functions, to provide the agent with an environment in which to act.

## Setup
Please install requirements from the `requirements.txt` file.

## Usage

The main flow is outlined in the `main.py` file, which expects one required argument (`--problem <id>`).
When run, it will load one of the files under `problems/` (the one with the number specified as an argument)
and will perform its task in the context of the loaded problem.

Example:
```
python main.py --problem 1
```

(Please run the command from the repo's main folder to avoid problems with imports).

Each problem is described and interacted with using the `Problem` object that is loaded from the relevant file
(each file has a global variable called `problem` that holds the objects).

The `Problem` object exposes the following main attributes/functions to use:
 - `fn` - the source python function
 - `cfn` - the python function compiled using the default numba flow
 - `cfn_src` - LLVM IR of the compiled function - make this part of the agent's input
 - `optimize(opt_ir)` - a method taking an optimized LLVM IR that is meant to replace `cfn_src`
    for the purpose of optimizing the workload - pass your agent's final output here
    > **Note:** The optimized IR should be a single function definition - it should start with "define ..."
    > (possibly preceeded with comments "; ...") and finish with a line containing "}"

    > **Note:** This function can raise errors!

 - `ai_cfn` - after `optimize(opt_ir)` has been called, the compiled function can be accessed via this property
 - `reset()` - can be used to call `optimize(opt_ir)` again
 - `get_test_data()` - returns a tuple of arguments that can be passed to a function for testing

See the `problems/api.py` file for details about the `Problem` class and `main.py` for some examples of using it.

> **Note:** If you are using VS Code, there is already a debug configuration present in the repo that launches the first problem in the debug console, for your convenience.

## Running the agent

Currently in the `main.py` there is a very simple `run_agent` function that acts as a stub for running the agent. This is the main integration point which should be used as a starting point.
In particular, look for the line `optimized = str(llvmir)` and replace it with your code that runs the agent.

Please do not modify any of the problem files and/or the compiler functions, unless it is strictly bug fixes or blockers for some higher-level functionality.
In particular, you should not make any optimizations to the compiled functions by the means of the compiler and/or problem definition themselves - this should only be attempted by running an AI agent in the `run_agent` method.

Having said that, you are otherwise allowed to do pretty much anything you want.
You might be interested, for example, in using the documentation of LLVM IR (https://llvm.org/docs/LangRef.html) and/or inspecting the original python function (https://stackoverflow.com/questions/427453/how-can-i-get-the-source-code-of-a-python-function). Keep in mind that the python function can be access as `Problem.fn`).


## How to approach it 

The purpose of this exercise is to gauge one's ability to build agentic systems that interact with highly technical stacks, such as compilers.
Consequently, you should not focus on the performance and/or quality of the final compiled program, but rather on the design of the system itself.
Do not worry if you are not familiar with LLVM IR or if your final code does not compile - the environment in this exercise is made complex on purpose,
since in real world we often need to interact with parts of the stack that we might not be familiar with (or have no control over).
Instead, focus on your part and do your best there!

Since the assignment is focused around system design, it is left open-ended on purpose and there are no particular tasks or objectives that we expect you to achieve
(there is no "one good answer" when it comes to system design).
However, you can consider (some) of the following steps/milestones as an example of what the progression could look like (you can also consider them to be functional requirements of the system):
 - make the agent explain what the input code does
 - make the agent explain what the main bottleneck in the input code is
 - make the agent suggest changes to the input IR and explain why they are supposed to help
 - make the agent improve (in some sense) over time with a feedback loop
Having said that, you are also free to ignore them if you think there is a better way of approaching the problem (be ready to explain why).

Importantly, while working please pay special attention to things such as:
 - determining how to prepare/postprocess the data that is fed to/obtained from the agent
 - determining the best way of providing the feedback to the agent
 - exploring different techniques to query the underlying LLM
 - making sure things do not break even if the agent does not produce valid programs
 - following good software engineering practices

On the other hand, the following things are not necessary, although they might be considered sufficient:
 - making the resulting program compile correctly
 - making the resulting program run faster than the baseline (the input to the agent)
 - fixing the provided code in any way

 Note that you can try to obtain as much information as possible from the internet to accomplish things.
 But please make sure to avoid any situations when the solution to the problem is given to the agent as part of its input.

Finally, for the sake of clarity: you are not expected to do all there is to be done here, but of course the more you achieve the better.

Good luck!
