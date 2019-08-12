import numpy as np
import matplotlib.pyplot as plt


# env = gym.make('FrozenLake-v0')   4*4
def plot(values, title='State-Value'):
    values = np.reshape(values, (4, 4))

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.imshow(values, cmap='cool')
    for (j, i), label in np.ndenumerate(values):
        ax.text(i, j, np.round(label, 5), ha='center', va='center', fontsize=14)
    plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.title(title)
    plt.savefig('./image/' + title + '.png')
    plt.show()


if __name__ == '__main__':
    x = np.random.random((4, 4))
    plot(x)
