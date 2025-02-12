from problems.api import Problem
import numpy as np


def matmul(m, k, n, A, B):
    out = np.zeros((m, k), dtype=np.float32)
    for i in range(m):
        for j in range(n):
            for p in range(k):
                out[i][j] += A[i][p] * B[p][j]

    return out


np.random.seed(12345)

A = np.asarray((np.random.rand(64, 32) * 4) - 2, dtype=np.float32)
B = np.asarray((np.random.rand(32, 16) * 4) - 2, dtype=np.float32)


problem = Problem(matmul, 'float32[:,:](int32, int32, int32, float32[:,:], float32[:,:])', (A.shape[0], A.shape[1], B.shape[1], A, B))
