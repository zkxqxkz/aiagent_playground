
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


class PromptParser(ABC):
    ...
    

class QuestionAnsweringParser(PromptParser):
    
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

       
        
@sgl.function
def run_agent(s, system_prompt: str, user_prompts: str, max_tokens: int = 256):

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
        