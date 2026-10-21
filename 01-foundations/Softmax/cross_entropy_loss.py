import numpy as np


def relative_entropy_loss(x, y):
    px = np.array(x)
    py = np.array(y)
    mask = px > 0   # 约定 0 * log(0) = 0，只对 px > 0 的项求和
    return np.sum(px[mask] * np.log(px[mask] / py[mask]))

def cross_entropy_loss(x, y):
    px = np.array(x)
    py = np.array(y)
    return -np.sum(px * np.log(py))


if __name__ == "__main__":
    px = [1, 0, 0]
    py = [0.1, 0.2, 0.7]
    print(relative_entropy_loss(px, py))
    print(cross_entropy_loss(px, py))