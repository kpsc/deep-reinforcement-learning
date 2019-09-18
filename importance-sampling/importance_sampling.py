import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def gaussian(x, u, sigma):
    return np.exp(-(x - u) ** 2 / (2 * sigma ** 2)) / np.sqrt(2 * np.pi * sigma ** 2)


def importance_sampling_test(ori_sigma, sample_sigma=[]):
    origin = []
    isample = [[] for _ in range(len(sample_sigma))]
    fun = lambda x: [xi.append(0) for xi in x]
    for _ in tqdm(range(100)):
        origin.append(0)
        fun(isample)
        for i in range(50000):
            a = np.random.normal(1.0, ori_sigma)
            origin[-1] += a

            na = gaussian(a, 1.0, ori_sigma)
            for k, sigma in enumerate(sample_sigma):
                ua = gaussian(a, 1.0, sigma)
                isample[k][-1] += a * min(na / ua, 1.0)

    origin = np.array(origin)
    isample = np.array(isample)

    print(np.mean(origin), np.std(origin))
    print(np.mean(isample), np.std(isample))


def plot():
    xs = np.linspace(-5, 6, 301)
    y1 = [gaussian(x, 1.0, 1.0) for x in xs]
    y2 = [gaussian(x, 1.0, 0.5) for x in xs]
    y3 = [gaussian(x, 1.0, 2.0) for x in xs]

    fig = plt.figure(figsize=(8, 5))
    plt.plot(xs, y1, label='sigma=1.0')
    plt.plot(xs, y2, label='sigma=0.5', linestyle=':')
    plt.plot(xs, y3, label='sigma=2.0', linestyle='--')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot()

    importance_sampling_test(1.0, 1.0)
    importance_sampling_test(1.0, 0.5)
    importance_sampling_test(1.0, 2.0)
