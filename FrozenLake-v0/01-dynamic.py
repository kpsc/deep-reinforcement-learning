import copy
import numpy as np
from utils import plot
from frozenlake import FrozenLakeEnv


params = {
    'epison': 0.1,  # random action
    'theta': 1e-8,  # converge threshold
    'gamma': 1.0   # discount reward
}


class Dynamic():
    def __init__(self, env, params):
        self.env = env
        self.epison = params.get('epison', 0.1)
        self.theta = params.get('theta', 1e-8)
        self.gamma = params.get('gamma', 0.999)

        self.num_actions = env.action_space.n
        self.num_states = env.observation_space.n

        self.actions = self.env.actions
        self.states = self.env.states

        self.V = np.zeros(self.num_states, dtype=np.float)
        self.policy = np.ones((self.num_states, self.num_actions), dtype=np.float) / self.num_actions

    def policy_evaluation(self, policy=None):
        if policy is None:
            policy = self.policy
        V = np.zeros_like(self.V)
        while True:
            delta = 0.0
            for state in self.states:
                Vs = 0.0
                for action, p_a in enumerate(policy[state]):
                    next_state, reward, done, _ = self.env.step(state, action)
                    Vs += p_a * (reward + self.gamma * V[next_state])
                delta = max(delta, abs(V[state] - Vs))
                V[state] = Vs
            if delta < self.theta:
                break

        self.V = V
        return V

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

    def policy_iteration(self):
        policy = copy.deepcopy(self.policy)
        while True:
            V = self.policy_evaluation(policy)
            new_policy = self.policy_improvement()

            # OPTION 1: stop if the policy is unchanged after an improvement step
            # if (new_policy == policy).all():
            #     break

            # OPTION 2: stop if the value function estimates for successive policies has converged
            error = np.max(abs(self.policy_evaluation(policy) - self.policy_evaluation(new_policy)))
            if error < self.theta*1e2:
               break

            policy = copy.deepcopy(new_policy)
        return policy, V

    def value_iteration(self):
        V = np.zeros_like(self.V)
        while True:
            delta = 0.0
            for state in self.states:
                Vs = 0.0
                for action, p_a in enumerate(self.policy[state]):
                    next_state, reward, done, _ = self.env.step(state, action)
                    vs = p_a * (reward + self.gamma * V[next_state])
                    if vs > Vs:
                        Vs = vs
                delta = max(delta, abs(V[state] - Vs))
                V[state] = Vs
            if delta < self.theta:
                break
        self.V = V
        policy = self.policy_improvement()

        return policy, V


if __name__ == '__main__':
    env = FrozenLakeEnv()
    model = Dynamic(env, params)
    # V = model.policy_evaluation(model.policy)
    # policy = model.policy_improvement()

    policy, V = model.value_iteration()
    plot(V, 'dynamic-value-iteration')

    env.eval(policy)
