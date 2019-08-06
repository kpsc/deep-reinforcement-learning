import os
import gym
import numpy as np
import tensorflow as tf
from model import ActorCritic


def enjoy(agent, env, render=False):
    states = env.reset()
    state_size = (1, 4)
    episode_reward = 0
    done = False
    while not done:
        if render:
            env.render()
        states = np.reshape(states, state_size)
        action = agent.get_action(sess, states)
        next_states, reward, done, info = env.step(action)
        episode_reward += reward
        states = next_states
    env.close()

    return episode_reward


def train(sess, env):
    done = False
    state_size = (1, 4)
    state = np.reshape(env.reset(), state_size)

    while not done:
        action = agent.get_action(sess, state)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, state_size)

        batch = [state, action, reward, next_state, done]
        output = agent.train_step(sess, batch)
        policy_loss = output['pg_loss']
        value_loss = output['vf_loss']

        state = next_state

    return policy_loss, value_loss


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    # env = gym.make('CartPole-v1')
    istrain = False

    with tf.Session() as sess:
        agent = ActorCritic(env)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=10)

        if istrain:
            episode_rewards = []
            for e in range(1, 5000):
                policy_loss, value_loss = train(sess, env)

                if e % 50 == 0:
                    reward = enjoy(agent, env)
                    print('episode: %d, reward: %d, pg_loss: %.4f, vf_loss: %.4f' %
                          (e, reward, policy_loss, value_loss))

                    episode_rewards.append(reward)
                    if np.mean(episode_rewards[-5:]) > 195:
                        if not os.path.exists('./logs/model-save/'):
                            os.makedirs('./logs/model-save/')
                        saver.save(sess, './logs/model-save/CartPole-v0.ckpt')
                        break
        else:
            try:
                saver.restore(sess, './logs/model-save/CartPole-v.ckpt')
                for i in range(5):
                    r = enjoy(agent, env, True)
                    print('Episode: %d,  reward: %d' % (i, r))
            except:
                print('Please give model which has been trained!')