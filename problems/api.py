import copy

import compiler as C
from cpuinfo import get_cpu_info

class Problem():
    def __init__(self, fn, signature, test_data):
        self.fn = fn
        self.signature = signature
        self.test_data = test_data

        self._cfn = None
        self._cfn_name = None # name of the function as present in the compiled module
        self._cfn_src = None # source LLVM-IR of the compiled function

        self._ai_cfn = None
        self._ai_src = None # AI-optimized LLVM-IR of the target function
        

        cpu_info = get_cpu_info()
        self.cpu_info  = (
            f"CPU: {cpu_info['brand_raw']} | "
            f"Architecture: {cpu_info['arch']} | "
            f"Bits: {cpu_info['bits']} | "
            f"Cores: {cpu_info['count']} | "
            f"L2 Cache: {cpu_info.get('l2_cache_size', 'Unknown')} | "
            f"L3 Cache: {cpu_info.get('l3_cache_size', 'Unknown')}"
        )

    def get_test_data(self):
        return copy.deepcopy(self.test_data)

    @property
    def cfn(self):
        ''' Compiled function (w/o AI) '''
        self.compile()
        return self._cfn

    @property
    def cfn_src(self):
        ''' Source LLVM-IR of the compiled function (w/o) AI '''
        self.compile()
        return self._cfn_src

    @property
    def ai_cfn(self):
        if self._ai_cfn is None:
            raise ValueError('AI-optimized function does not exist. Make sure to call .optimize() before requesting it')
        return self._ai_cfn

    def compile(self):
        if self._cfn is None:
            self._cfn, self._cfn_name, self._cfn_src = C.compile(copy.deepcopy(self.fn), self.signature)

    def optimize(self, ai_src):
        ''' Replace original LLVM IR with a new version (generated by the agent) '''
        if self._ai_cfn is not None:
            raise RuntimeError('Already optimized! Call .reset() if you want to recompile')

        self.compile()

        self._ai_src = ai_src
        self._ai_cfn = C.optimize(copy.deepcopy(self.fn), self.signature, self._cfn_name, self._ai_src)

    def reset(self):
        ''' Resets the optimized version '''
        self._ai_cfn = None
        self._ai_src = None
