import contextlib
import platform
import re

import llvmlite.binding as ll

from numba import njit
from numba.core import utils, externals, registry, cgutils
from numba.core.target_extension import cpu_target, dispatcher_registry
from numba.core.compiler_lock import global_compiler_lock
from numba.core.codegen import JITCPUCodegen, JITCodeLibrary
from numba.core.cpu import CPUContext
from numba.core.utils import threadsafe_cached_property as cached_property
from numba.core.registry import CPUDispatcher


_target_fn_name = None
_target_fn_module = None
_cfn_name = None
_cfn_src = None
_cfn_opt_llvmir = None
_found = False


def _getstate():
    return (_target_fn_name, _target_fn_module, _cfn_name, _cfn_src, _cfn_opt_llvmir, _found)


def _setstate(state):
    global _target_fn_name, _target_fn_module, _cfn_name, _cfn_src, _cfn_opt_llvmir, _found
    _target_fn_name, _target_fn_module, _cfn_name, _cfn_src, _cfn_opt_llvmir, _found = state


def _resetstate():
    _setstate([None for _ in _getstate()])


class AiTargetFn():
    def __init__(self, module, fn_name):
        self.module = module
        self.fn_name = fn_name

        self._module_src = None
        self._target_fn = None
        self._fn_src = None
        self._opt_fn_src = None
        self._opt_fn_name = None
        self._opt_module = None

    @property
    def module_src(self):
        ''' Get IR of the whole source module '''
        if self._module_src is None:
            self._module_src = str(self.module)
        return self._module_src

    @property
    def target_fn(self):
        ''' Get target function object or None, if doesn't exist '''
        if self._target_fn is None:
            for fn in self.module.functions:
                if fn.name == self.fn_name:
                    self._target_fn = fn
                    break

        return self._target_fn

    @property
    def fn_src(self):
        ''' Get IR of the target function if exists, or None '''
        if self._fn_src is None:
            fn = self.target_fn
            if fn is not None:
                self._fn_src = str(fn)

        return self._fn_src

    def optimize(self, opt_ir, fn_name):
        if self._opt_module is not None:
            raise ValueError('Module already optimized!')
        if self.target_fn is None:
            print('[WARNING] Called .optimize() on a LLVM-IR function target that could not resolved, the call will be ignored')
            return

        opt_ir = opt_ir.strip()
        nonempty = False
        for line in opt_ir.splitlines():
            if line.startswith(';'):
                continue
            nonempty = True
            if not line.startswith('define '):
                raise ValueError(f'Optimized function IR should start with a function definition ("define ...")! But got: {opt_ir[:15]!r}')

            break

        if not nonempty:
            raise ValueError('Empty optimized IR!')
        
        if not opt_ir.endswith('\n}'):
            raise ValueError('Optimized IR should end with a newline followed by a closing brace "}"')

        self._opt_fn_src = opt_ir
        new_mod_src = self.patch_source_ir(self.module_src, self.fn_name, opt_ir, fn_name)
        self._opt_module = ll.parse_assembly(new_mod_src)
        self._opt_module.name = cgutils.normalize_ir_text(self.module.name)

    @property
    def final_module(self):
        return self._opt_module if self._opt_module else self.module

    @staticmethod
    def patch_source_ir(src, fn_name, patched_fn, patched_fn_name):
        # src has a function called "fn_name" and patched_fn is a definition
        # of a function called "patched_fn_name" (supposedly);
        # we are going to insert the patched_fn into src with using name "fn_name"
        # but arguments and return type that of "patched_fn"

        source_fn_def_re = re.compile(r'(; Function Attrs: (.*?)\n+)?define (.*?) @' + fn_name + r'(\(.*\) .*) \{', re.MULTILINE)
        target_fn_def_re = re.compile(r'(; Function Attrs: (.*?)\n+)?define (.*?) @' + patched_fn_name + r'(\(.*\) .*) \{', re.MULTILINE)

        target_fn_def = target_fn_def_re.search(patched_fn)
        assert target_fn_def is not None
        assert target_fn_def.start() == 0

        new_fn_def = ''
        if target_fn_def.group(1):
            new_fn_def = target_fn_def.group(1)
        new_fn_def += 'define ' + target_fn_def.group(3) + ' @' + fn_name + target_fn_def.group(4) + ' {'

        new_src = ''

        source_def = source_fn_def_re.search(src)
        if not source_def:
            raise ValueError('Could not find source function to replace!')

        new_src = src[:source_def.start()]
        new_src += new_fn_def
        new_src += patched_fn[target_fn_def.end():] + '\n'

        cut = True
        for line in src[source_def.end():].splitlines():
            if not cut:
                new_src += line + '\n'
            else:
                if line == '}':
                    cut = False

        return new_src


class AiCodeLibrary(JITCodeLibrary):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._source_module = None
        self._ai_target = None

    def _optimize_final_module(self):
        super()._optimize_final_module()

        global _cfn_name, _cfn_src, _found

        ai_target_fn = self._find_ai_target_fn(self._final_module)
        if ai_target_fn is not None:
            if _found:
                raise RuntimeError('Multiple target functions found!')
            _found = True

            _ai_target = AiTargetFn(self._final_module, ai_target_fn.name)
            if _cfn_name is None:
                _cfn_name = _ai_target.fn_name
                _cfn_src = _ai_target.fn_src

            if _cfn_opt_llvmir:
                _ai_target.optimize(_cfn_opt_llvmir, _cfn_name)

            self._final_module = _ai_target.final_module

    def _find_ai_target_fn(self, module):
        if module.name != _target_fn_name:
            return None

        ret = None
        required = False
        if _cfn_name:
            required = True

        mangled_fn_fullname = ''
        if _target_fn_module:
            for part in _target_fn_module.split('.'):
                mangled_fn_fullname += str(len(part)) + part
        mangled_fn_fullname += str(len(_target_fn_name)) + _target_fn_name

        target_fn_re = re.compile(r'^_ZN' + mangled_fn_fullname)

        ret = None
        for fn in module.functions:
            if target_fn_re.search(fn.name):
                if ret is not None:
                    raise RuntimeError('More than one target function found!')
                ret = fn

        if required and ret is None:
            raise ValueError(f'Could not find a compiled function called: {_cfn_name!r}')

        return ret


class AiCodegen(JITCPUCodegen):
    _library_class = AiCodeLibrary


class AiContext(CPUContext):
    @global_compiler_lock
    def init(self):
        self.is32bit = (utils.MACHINE_BITS == 32)
        self._internal_codegen = AiCodegen("numba.exec")

        # Add ARM ABI functions from libgcc_s
        if platform.machine() == 'armv7l':
            ll.load_library_permanently('libgcc_s.so.1')

        # Map external C functions.
        externals.c_math_functions.install(self)


class AiTarget(registry.CPUTarget):
    @cached_property
    def _toplevel_target_context(self):
        # Lazily-initialized top-level target context, for all threads
        return AiContext(self.typing_context, self._target_name)


ai_target = AiTarget('cpu')


class AiDispatcher(CPUDispatcher):
    targetdescr = ai_target


@contextlib.contextmanager
def override_dispatcher():
    old = dispatcher_registry.pop(cpu_target)
    dispatcher_registry[cpu_target] = AiDispatcher
    try:
        yield
    finally:
        dispatcher_registry.pop(cpu_target)
        dispatcher_registry[cpu_target] = old


def compile(func, signature):
    state = _getstate()
    _resetstate()

    global _target_fn_name, _target_fn_module

    _target_fn_name = func.__name__
    _target_fn_module = func.__module__

    try:
        with override_dispatcher():
            cfn = njit(signature, cache=False)(func)
            if not _found:
                raise ValueError('Could not find the target function!')
            cfn_name = _cfn_name
            cfn_src = _cfn_src
    finally:
        _setstate(state)

    return cfn, cfn_name, cfn_src


def optimize(func, signature, cfn_name, opt_llvmir):
    state = _getstate()
    _resetstate()

    global _target_fn_name, _target_fn_module, _cfn_name, _cfn_opt_llvmir

    _target_fn_name = func.__name__
    _target_fn_module = func.__module__
    _cfn_name = cfn_name
    _cfn_opt_llvmir = opt_llvmir

    try:
        with override_dispatcher():
            cfn = njit(signature, cache=False)(func)
            if not _found:
                raise ValueError('Could not find the target function!')
    finally:
        _setstate(state)

    return cfn
