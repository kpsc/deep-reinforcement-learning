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


class MonteCarlo():
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
        self.policy = np.ones((self.num_states, self.num_actions), dtype=np.float) / self.num_actions

    def policy_improvement(self):
        policy = np.zeros_like(self.policy)
        for state in self.states:
            value = np.zeros_like(self.actions, dtype=np.float)
            for action in self.actions:
                next_state, reward, _, _ = self.env.step(state, action)
                value[action] = reward + self.gamma * self.V[next_state]

            best_a = np.argwhere(value == np.max(value)).flatten()
            policy[state] = np.sum([np.eye(self.num_actions)[i] for i in best_a], axis=0) / len(best_a)

        self.policy = policy

        return policy

    def generate_episode(self):
        state = self.env.reset()
        done = False

        episode = []
        while not done:
            action = self.get_action(state, self.policy)
            next_state, reward, done, _ = self.env.step(state, action)

            episode.append([state, action, reward])
            state = next_state

        return episode

    def get_action(self, state, policy):
        if np.random.random() < self.epison:
            return np.random.choice(self.actions)
        else:
            p = policy[state]
            p = np.exp(p) / np.sum(np.exp(p))
            return np.random.choice(self.actions, p=p)

    def first_visit(self):
        for _ in tqdm(range(0, int(1e3))):
            episode = self.generate_episode()
            G = 0.0
            visit_state = []    # remove this for every visit
            for (state, action, reward) in reversed(episode):
                if state not in visit_state:
                    visit_state.append(state)
                    G = reward + self.gamma * G
                    self.V[state] = self.V[state] + self.alpha * (G - self.V[state])

            self.policy = self.policy_improvement()

            if episode[-1][-1] > 0:
                print(_)

        return self.policy, self.V


if __name__ == '__main__':
    env = FrozenLakeEnv()
    model = MonteCarlo(env, params)
    policy, V = model.first_visit()
    plot(V, 'montecarlo-first-visit')

    env.eval(policy)
