
from typing import List, Union
from abc import ABC, abstractmethod
from enum import Enum
import sglang as sgl
from loguru import logger
from sglang.lang.interpreter import ProgramState

class AgentType(Enum):
    question_answering = "question_answering"
    correctness = "correctness"
    optimization = "optimization"

class QuestionAnsweringParser:
    
    def __init__(self):
        
        self.system_prompt =  """
        You are the best numba njit LLVM IR developer providing clear and detailed information.
        # Given the IR code you are able to check the correctness, expected run-time, expected memory usage, bottlenecks, best use-case and worst use-case.
        
        """
        self.user_prompt = """
        You are given a numba njit compiled code below:
        {}
        
        Your task is to answer the following prompt:
        {}
        
        """
        
    def parse_to_user_prompt(self, ir_code: str, questions: List[str]):
        return [
            self.user_prompt.format(ir_code, question)
            for question in questions
        ]
        

class QuestionAnsweringAgent(QuestionAnsweringParser):
    
    type = AgentType.question_answering

    def __init__(self, model:str):
        super().__init__()
        
        self.state = None
        
        self.runtime = sgl.Runtime(model_path=model)
        sgl.set_default_backend(self.runtime)
        
    def __call__(
        self,
        ir_code: str,
        questions: Union[str, List[str]],
    ):
        if isinstance(questions, str):
            questions = [questions]
        
        user_prompts: List[str] = self.parse_to_user_prompt(ir_code, questions)
        
        self.state: ProgramState = run_agent(self.system_prompt, user_prompts)
        return self.state
    
    def shutdown(self):
        self.runtime.shutdown()
        
    def parse_output(self):
        if self.state is not None:
            for m in self.state.messages():
                print("================================================")
                print(m["role"], ":", m["content"])
                print("================================================")

       
class OptimizationPlanningParser:
    def __init__(self,):
        
        # provide {machine} 
        self.system_prompt = """
        You are the best numba njit LLVM IR developer who can assess the provided LLVM IR code with its correctness, expected run-time and expected memory usage. 
        
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
        self.user_prompt = """
        
        The plan must include in detail:
            1. Explainations on the current bottleneck of the code with respect to the machine that will run the code.
            2. Explainations on the current bottleneck of the code with respect to the operations in the given IR code. 
            3. Explainations on what can be optimized, and what you will do.
            4. Explanations on why the planned optimization is correct and will be able to run faster than the original IR code.
        
        Once the above is answered, make a summary of changes, starting with "CHANGES TO BE MADE:".
        
        Please ONLY include information on how the LLVM IR code will be optimized, everything else is irrelevant.
        """
        
class OptimizationPlanningAgent(OptimizationPlanningParser):

    def __init__(self, model:str):
        super().__init__()
        
        self.state = None
        
        self.runtime = sgl.Runtime(model_path=model)
        self.plan = None
        sgl.set_default_backend(self.runtime)
        
    def __call__(
        self,
        *args,
    ):
        assert len(args) == 6
        self.state: ProgramState = run_agent(
            self.system_prompt.format(*args), 
            [self.user_prompt]
        )
        return self.state
    
    def shutdown(self):
        self.runtime.shutdown()
        
    def parse_output(self):
        if self.state is not None:
            for m in self.state.messages():
                print("================================================")
                print(m["role"], ":", m["content"])
                print("================================================")

    def genrate_plan(self):
        system_prompt = """
        You are an top information extractor.
        """
        
        user_prompt = f"Extract me the overall plan. It starts with 'CHANGES TO BE MADE:`. Only provide me what is in the given prompt, do not add new content.'\n\n {self.state.text}"
        
        state = run_agent(system_prompt, [user_prompt], max_tokens=256)
        self.plan = state.messages()[-1]["content"]
        return state.messages()[-1]["content"]
        
    

@sgl.function
def run_agent(s, system_prompt: str, user_prompts: str, max_tokens: int = 1024):

    s += sgl.system(system_prompt)
    for prompt in user_prompts:
        s += sgl.user(prompt)
        s += sgl.assistant(sgl.gen(f"answer", max_tokens=max_tokens))
        
        
    
    
class CheckCorrectnessAgent:
    ...
    
class CodeOptimizationAgent:
    ...
    
    
class LazyAgentInitializer:
    """
    LazyLoad Agents based on need
    """
    
    factory = {
        AgentType.question_answering:   QuestionAnsweringAgent,
        AgentType.correctness:          CheckCorrectnessAgent,
        AgentType.optimization:         CodeOptimizationAgent,
    }
    
    initialized = {}
    
    def get(self, agent_type: str, **kwargs):
        
        agent = None
        if agent_type in AgentType.__members__:
            _type = AgentType[agent_type]
            if self.initialized.get(_type) is not None:
                return agent
            
            agent = self.factory[_type](**kwargs)
            self.initialized[_type] = agent
                
            return agent

        logger.warning(f"Agent type {agent_type} not found")
        
    
    
    
# class Agent:
    
#     agents = {
#         "question_answering": QuestionAnsweringAgent,
#         "correctness": CheckCorrectnessAgent,
#         "optimization": CodeOptimizationAgent, 
#     }
    
#     def __init__(self, model: str):
#         self.model = model
        
#     def __call__(self):
#         # raise NotImplementedError
        
#         # agent, prompts = 
    
#     # @abstractmethod
#     # def run_agent(self):
#     #     # raise NotImplementedError
#     #     ...
        