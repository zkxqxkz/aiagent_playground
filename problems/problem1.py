from problems.api import Problem
import numpy as np


def foo(x):
    return np.exp(-x) * np.sin(x)


problem = Problem(foo, 'float64(float64)', (np.float64(0.12345), ))
