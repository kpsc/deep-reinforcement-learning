import numpy as np


def enjoy(sess, env, agent, render=False):
    states = env.reset()
    state_size = (1, 4)
    episode_reward = 0
    done = False
    while not done:
        if render:
            env.render()
        states = np.reshape(states, state_size)
        action = agent.get_action(sess, states, False)
        next_states, reward, done, info = env.step(action)
        episode_reward += reward
        states = next_states
    env.close()

    return episode_reward


def generate_episode(sess, env, agent):
    episode = []

    done = False
    state_size = (1, 4)
    state = env.reset()

    while not done:
        state = np.reshape(state, state_size)
        action = agent.get_action(sess, state)
        next_state, reward, done, info = env.step(action)

        if done:
            reward = -5
        episode.append([state, action, reward])

        state = next_state

    return episode
