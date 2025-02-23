PLANNING_SYSTEM_PROMPT = """
You are the best numba njit LLVM IR developer who can assess the provided LLVM IR code with its correctness, and the expected run-time.

Given 
    1. njit LLVM IR code, compiled from python code using numba jit:
    {}
    2. Original Python code used to create the njit LLVM IR code:
    {}
    3. The test data that will be used as the input to the generated code:
    {}
    4. The output from from the llvm code and the test data
    {}
    5. Run-time of the provided LLM IR Code with the provided test input:
    {}
    6. The machine that will run the code:
    {}

Create plan to optimize the the given LLM IR Code. Only focus on optimizing the IR code, others are only used as reference or validation that the output is correct.
You must ensure that the provided LLVM IR output code is correct.
You must optimize the run-time, ideally faster than the originally provided code
You may refer to the LLVM docs (https://llvm.org/docs/LangRef.html).

"""

PLANNING_USER_PROMPT = """

The plan must include in detail:
    1. Explainations on the current bottleneck of the code with respect to the machine that will run the code.
    2. Explainations on the current bottleneck of the code with respect to the operations in the given IR code. 
    3. Explainations on what can be optimized, and what you will do.
    4. Explanations on why the planned optimization is correct and will be able to run faster than the original IR code.

Once the above is generated, summarize it starting with "CHANGES TO BE MADE:".
Please ONLY include information on how the LLVM IR code will be optimized.
"""


OPTIMIZATION_SYSTEM_PROMOPT = """
You are the best numba njit LLVM IR optimization developer. 
Your task is to optimize the given njit LLVM IR code, compiled from python code using numba jit

{}
"""

CODE_GEN_USER_PROMPT = """
Use the following plan to optimize for runtime:
{}

Please make sure that the given the test data: {}, the output is {} with the optimized code.

Remember that you must:
    - You are generating code to optimize the given LLVM IR code only.
    - Do not stray from the plan we have decided on.
    - MOST IMPORTANT OF ALL - every line of code you generate must be valid code. Do not include code fences in your response.

The generated code must satisfy:
    - Optimized code must run faster than the given IR code.
    - Optimized function IR MUST start with a Function Attributes `; Function Attrs:` then the function definition (`define ...`). These must be the first two lines of the generated code
    - Optimized IR MUST end with a newline followed by a closing brace. This should be the last characters of the generation.
    - No declarations (declare) or extra lines should follow after the function definition.
    - It must have just one `; Function Attrs:`
    - The function name must be `@{}`
    - Do not include any code fense, just output the code

"""
CODE_GEN_USER_PROMPT_RESPONSES = """

Example of a good response. 

    ; Function Attrs: ...
    define ... @... {
        ...
    }\n

    Starts with `; Function Attrs ...` then `define` with the specific name, and then a closing with newline with no code after.
    Also does not have code fense

Example of a bad response:
    ```llvm
    ; Function Attrs: ...
    define ... @...{
        ...
    }

    ; Function Attrs: nounwind readnone speculatable
    declare double @llvm.exp.f64(double) #1

    ; Function Attrs: nounwind readnone speculatable
    declare double @llvm.sin.f64(double) #1
    ```
    This is bad because 1. it does not end with a closing bracket, 2. more than one Function Attrs are defined. It must just contain one and 3. has a code fense
    
    ```
    ; Function Attrs: ...
    declare ... @...

    ; Function Attrs: ...
    define ... @... {
        ...
    }
    ```
    This is bad because more than one Function Attrs are defined. It must just contain one. It also has code fense which is invalid

Please make sure that the above format is respected. This is the only acceptable format.
    
"""
    
