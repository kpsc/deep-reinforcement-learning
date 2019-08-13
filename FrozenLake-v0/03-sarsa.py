import copy
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from utils import plot
from frozenlake import FrozenLakeEnv


params = {
    'epison': 0.1,  # random action
    'theta': 1e-8,  # converge threshold
    'gamma': 0.99,   # discount reward
    'alpha': 0.01
}


class Sarsa():
    def __init__(self, env, params):
        self.env = env
        self.epison = params.get('epison', 0.1)
        self.theta = params.get('theta', 1e-8)
        self.gamma = params.get('gamma', 0.999)
        self.alpha = params.get('alpha', 0.01)

        self.num_actions = env.action_space.n
        self.num_states = env.observation_space.n

        self.actions = self.env.actions
        self.states = self.env.states

        self.V = np.zeros(self.num_states, dtype=np.float)
        self.Q = np.zeros((self.num_states, self.num_actions), dtype=np.float)
        self.policy = np.ones((self.num_states, self.num_actions), dtype=np.float) / self.num_actions

    def get_policy(self):
        policy = np.zeros_like(self.policy)
        for state in self.states:
            value = self.Q[state]
            best_a = np.argwhere(value == np.max(value)).flatten()
            policy[state] = np.sum([np.eye(self.num_actions)[i] for i in best_a], axis=0) / len(best_a)

        self.policy = policy

        return policy

    def get_action(self, state, eps):
        if np.random.random() < eps:
            return np.random.choice(self.actions)
        else:
            p = self.Q[state]       # when I use np.argmax(self.Q[state]), can't achieve convergence
            p = np.exp(p) / np.sum(np.exp(p))
            return np.random.choice(self.actions, p=p)

    def q2v(self):
        # q = sum[R], s, a
        # v = sum[R], s
        # q = np.argmax(r + v(s')) for all(s')
        # v = sum(p_a * R(s,a)) for all(a)
        for s, p, q in zip(self.states, self.policy, self.Q):
            self.V[s] = np.sum(p * q) * 1e3

        return self.V

    def __call__(self, steps=1000):
        for step in tqdm(range(1, steps+1)):
            eps = max(1.0 / step, 0.01)

            state = self.env.reset()
            action = self.get_action(state, eps)
            done = False
            while not done:
                next_state, reward, done, _ = self.env.step(state, action)
                next_action = self.get_action(state, eps)
                Q_ = 0.0 if done else self.Q[next_state][next_action]
                self.Q[state][action] = self.Q[state][action] + self.alpha*(reward + self.gamma * Q_ - self.Q[state][action])

                state, action = next_state, next_action

                # if reward > 0:      # when we get reward, self.Q was changed slowly
                #     print(step)

            self.policy = self.get_policy()

        return self.policy, self.q2v()


if __name__ == '__main__':
    env = FrozenLakeEnv()
    model = Sarsa(env, params)
    policy, V = model()
    plot(V, 'sarsa v*1e3')

    env.eval(policy)
