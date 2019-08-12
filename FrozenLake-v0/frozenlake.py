import numpy as np
import sys
from six import StringIO, b

import gym
from gym import utils
from gym.envs.toy_text import discrete

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],
}


class FrozenLakeEnv():
    def __init__(self, map='4x4'):
        self.map = np.asarray(MAPS[map], dtype='c')
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Discrete(self.map.size)

        self.nrow, self.ncol = self.map.shape

        self.actions = np.arange(self.action_space.n, dtype=np.int)
        self.states = np.arange(self.map.size, dtype=np.int)

        self.state = 0
        self.lastaction = None

    def reset(self):
        self.state = 0

        return self.state

    def step(self, s, a):
        self.lastaction = a
        row, col = s // self.ncol, s % self.ncol
        if self.map[row][col] in b'GH':
            return s, 0, True, None

        if a == 0:  # left
            col = max(0, col - 1)
        elif a == 1:  # down
            row = min(row+1, self.nrow-1)
        elif a == 2:  # right
            col = min(col+1, self.ncol-1)
        elif a == 3:  # up
            row = max(row-1, 0)

        s_new = row * self.ncol + col

        if self.map[row][col] == b'G':
            reward = 1
        # elif self.map[row][col] == b'H':
        #     reward = -1
        else:
            reward = 0

        done = False
        if self.map[row][col] in b'GH':
            done = True

        self.state = s_new

        return s_new, reward, done, None

    def render(self):
        outfile = sys.stdout
        row, col = self.state // self.ncol, self.state % self.ncol
        map = self.map.tolist()
        map = [[c.decode('utf-8') for c in line] for line in map]
        map[row][col] = utils.colorize(map[row][col], 'red', highlight=True)

        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left", "Down", "Right", "Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write('\n'.join(''.join(line) for line in map) + '\n')

    def eval(self, policy):
        s = self.reset()
        done = False

        self.render()
        while not done:
            a = np.argmax(policy[s])
            s, r, done, _ = self.step(s, a)
            self.render()

        if s == self.states[-1]:
            print('Complete!')
            return True
        else:
            print('Game Over!')
            return False


if __name__ == '__main__':
    env = FrozenLakeEnv()
    env.render()
