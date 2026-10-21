import numpy as np

def softmax(x):
    p_x = np.exp(x)
    return p_x / np.sum(p_x)


def softmax_stable(x):
    x = np.asarray(x)
    x_shifted = x - np.max(x)
    p_x = np.exp(x_shifted)
    return p_x / np.sum(p_x)



def logsoftmax(x):
    x = np.asarray(x)
    x_shifted = x - np.max(x)
    return x_shifted - np.log(np.sum(np.exp(x_shifted)))


def softmax_T(x, T = 1.0):
    x = np.asarray(x)
    x_shifted = x - np.max(x)
    p_x = np.exp(x_shifted / T)
    return p_x / np.sum(p_x)
    

if __name__ == "__main__":
    logtist = [10, 2, 1000]
    # print(np.log(softmax(logtist)))
    # print(softmax_stable(logtist))
    # print(logsoftmax(logtist))
    # print(softmax_T(logtist))
    # print(softmax_T(logtist, T = 0.1))
    # print(softmax_T(logtist, T = 10))

    logtist2 = [1, 3, 7]
    # print(softmax(logtist2))
    # print(softmax_stable(logtist2))
    # print(logsoftmax(logtist2))
    print(softmax_T(logtist2))
    print(softmax_T(logtist2, T = 0.1))
    print(softmax_T(logtist2, T = 10))

