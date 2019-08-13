import numpy as np
import matplotlib.pyplot as plt


# env = gym.make('FrozenLake-v0')   4*4
def plot(values, title='State-Value'):
    size = values.size
    nrow = int(np.sqrt(size))
    ncol = size // nrow
    assert nrow * ncol == size

    values = np.reshape(values, (nrow, ncol))

    fig = plt.figure(figsize=(nrow+2, ncol+2))
    ax = fig.add_subplot(111)
    ax.imshow(values, cmap='cool')
    for (j, i), label in np.ndenumerate(values):
        ax.text(i, j, np.round(label, 5), ha='center', va='center', fontsize=14)
    plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.title(title)
    plt.savefig('./image/' + title + '.png')
    plt.show()


if __name__ == '__main__':
    x = np.random.random(16)
    plot(x)
