import os
import gym
import numpy as np
import tensorflow as tf
from model import Reinforce
from utils import enjoy, generate_episode


def train(sess, env, agent, gamma=0.99):
    episode = generate_episode(sess, env, agent)

    loss = 0
    G = 0
    for s, a, r in reversed(episode[:-1]):  # with [:-1], converge fast...
        G = G*gamma + r
        loss = agent.train_step(sess, [s, a, G])

    return loss, len(episode) - 1


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    istrain = False

    with tf.Session() as sess:
        agent = Reinforce(env)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=10)

        if istrain:
            episode_rewards = []
            for e in range(1, 1000):
                loss, score = train(sess, env, agent)

                if e % 10 == 0:
                    reward = enjoy(sess, env, agent)
                    print('episode: %d, reward: %d, loss: %.4f, train-score: %.4f' %
                          (e, reward, loss, score))

                    episode_rewards.append(reward)
                    if len(episode_rewards) > 4 and np.mean(episode_rewards[-4:]) > 495:
                        if not os.path.exists('./logs/model-save/'):
                            os.makedirs('./logs/model-save/')
                        saver.save(sess, './logs/model-save/CartPole-v1.ckpt')
                        break

        else:
            try:
                saver.restore(sess, './logs/model-save/CartPole-v1.ckpt')
                for i in range(5):
                    r = enjoy(sess, env, agent, False)
                    print('Episode: %d,  reward: %d' % (i, r))
            except:
                print('Please give model which has been trained!')
