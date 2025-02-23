from typing import List, Dict
from openai import OpenAI

from .constants import *
import re
from typing import Optional


import os
from dotenv import dotenv_values
from .utils import completion_call


@completion_call
def extract_llvm_function(ir_code: str) -> str:
    """
    Genreate the optimized LLVM IR code that 
        start with a Function Attributes `; Function Attrs:`
        then a function definition ("define ...")
        ends with a single closing brace '}'.
        
    Example: 
    ```
    ; Function Attrs: ...
    define ... {
        ...
    } 
    ```
    
    """
    return ir_code  

class OAIAgent:
    def __init__(self, model: str, env: str = ".env"):
        self.model = model
        config = dotenv_values(env)

        self.client = OpenAI(
            api_key=config.get("OPENAI_API_KEY"),
        )
        self.optimized_codes = []

    def create_plan(
        self, 
        llvm_code: str, 
        python_code: str,
        test_data: float,
        output: float,
        runtime: float,
        cpu_info: str,
        save_path: str,
        cfn_name: str, 
        failed_plan: Optional[str] = None,
        failed_generation: Optional[str] = None,
        error: Optional[str] = None,
    ):
        print("Creating plan ...")
        messages = [
            {
                "role": "system",
                "content": PLANNING_SYSTEM_PROMPT.format(
                    llvm_code,
                    python_code,
                    test_data,
                    output,
                    runtime,
                    cpu_info,
                    cfn_name,
                ),
            },
            {
                "role": "user",
                "content": PLANNING_USER_PROMPT,
            },
        ]

        if failed_plan is not None:
            messages.append(
                {
                    "role": "user",
                    "content": f"The generated plan :\n{failed_plan}\n produces incorrect generation. Given the test data {test_data}, the plan does not produce the correct output: {output}",
                },
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"The generated code is incorrect:\n{failed_generation}\n"
                            f"Figure out what may have gone wrong and please produce the same output given this test data {test_data} that produces the output {output}.",
                },
            )

        if error is not None:
            messages.append(
                {
                    "role": "user",
                    "content": f"The generated code is incorrect:\n{failed_generation}\n"
                            f"Figure out what may have gone wrong and please produce the same output given this test data {test_data} that produces the output {output}. "
                            f"The produced error is: {error}",
                },
            )


        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        

        content = self.make_request(messages=messages,)

        with open(save_path, "w") as f:
            print(f"Wrting plan to {save_path}")
            f.write(content)
            
    def optimize(
        self, 
        path: str, 
        llvm_code: str,
        cfn_name: str,
        test_data: float,
        output: float,
        runtime: float,
    ):
        print("Execting plan...")
        plan = None
        with open(path, "r", encoding="utf-8") as file:
            plan = file.read()
            
        if plan is None:
            raise ValueError("No plan found")
        
        # only include the changes to apply
        pattern = r"CHANGES TO BE MADE[\s\S]*"
        match = re.search(pattern, plan)
        plan = match.group(0) if match else plan
        
        print(plan)
        
        messages = [
            {
                "role": "system",
                "content": OPTIMIZATION_SYSTEM_PROMOPT.format(llvm_code),
            },
            {
                "role": "user",
                "content": CODE_GEN_USER_PROMPT.format(plan, test_data, output, cfn_name) + CODE_GEN_USER_PROMPT_RESPONSES,
            },
        ]
        
        function_call = {"name": "extract_llvm_function"}
        functions = [extract_llvm_function.schema]

        response = self.make_request(
            messages=messages,
            # function_call=function_call,
            # functions=functions,
        )
        return response
            
    def make_request(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs,
    ):
        print(
            "OAI API call::Params: ",
            f"model: {self.model}",
            f"temperature: {temperature}",
            f"message keys: {list(messages[0].keys())}",
            f"params: {list(kwargs.keys())}",
        )
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=messages,
            stream=stream,
            **kwargs,
        )

        if "function_call" in kwargs:
            func_name = kwargs.get("function_call")["name"]
            # Get the function object from the string
            func = globals()[func_name]
            return func.from_response(response)

        message = response.choices[0].message
        return message.content