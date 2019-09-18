import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def deltax(x, theta=0.01, mu=0, sigma=0.02):
    wt = np.random.randn()
    dx = theta * (mu - x) + sigma * wt

    return dx


def generate(x=0.0, n=10000):
    s = [x]
    for _ in range(1, n):
        x += deltax(x)
        s.append(x)

    return s


def plot(data):
    t = list(range(len(data[0])))
    plt.figure() 
    for d in data:
        plt.plot(t, d)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    s0 = generate(x=0.0)
    s2 = generate(x=2.0)
    s_2 = generate(x=-2.0)

    data = [s0, s2, s_2]
    plot(data)
