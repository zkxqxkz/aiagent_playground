from problems.api import Problem
import numpy as np


def sort(a):
    for i in range(len(a)):
        for j in range(i+1, len(a)):
            if a[j] < a[i]:
                a[j], a[i] = a[i], a[j]

    return a


problem = Problem(sort, 'int64[:](int64[:])', (np.asarray([1, 21, -12, 94, 91, 0, 123, 4, 4, -1293, 9]), ))
