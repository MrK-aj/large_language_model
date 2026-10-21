import numpy as np
import os 

def mp(X, W):
    # 定义数组
    B, N, D = X.shape
    D2, M = W.shape
    assert D == D2

    Y = np.zeros((B, N, M))

    for b in range(B):
        for n in range(N):
            for m in range(M):  #m,决定当前计算输出元素位置
                for d in range(D): #d,遍历所有特征量，把乘积累加到当前输出元素中
                    Y[b, n, m] += X[b, n, d] * W[d, m]
    return Y

def mp2(X, W):
    B, N, D = X.shape
    M = W.shape[1]
    Y = np.zeros((B, N, M))

    for b in range(B):
            for n in range(N):
                for d in range(D):  #m,决定当前计算输出元素位置
                    Y[b, n, :] += X[b, n, d] * W[d, :]
    return Y

# einsum表示
def mp3(X, W):
    Y1 = X @ W
    Y2 = np.einsum('bnd, dm->bnm', X, W)
    print(Y1)
    print(Y2)
    print(np.allclose(Y1, Y2))

if __name__ == "__main__":
    X = np.array([[[1, 2, 3],
                 [4, 5, 6]]])

    W = np.array([[1, 10],
                   [2, 20],
                   [3, 30]])

    print(mp(X, W))
    print(mp2(X, W))
    mp3(X, W)

