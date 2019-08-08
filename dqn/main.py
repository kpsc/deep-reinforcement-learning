import os
import gym
import numpy as np
import tensorflow as tf
from model import DQN
from utils import LinearSchedule
from replay_buffer import ReplayBuffer


params = {
    'batch_size': 64,
    'lr': 0.001,
    'total_timesteps': 100000,
    'buffer_size': 50000,
    'exploration_fraction': 0.1, 
    'exploration_final_eps': 0.02,
}

learning_starts = 1000
target_network_update_freq = 500
print_freq = 100
train_freq = 1


def enjoy(sess, agent, env):
    states = env.reset()
    episode_reward = 0
    while True:
        env.render()
        action = agent.get_action(sess, states, False)
        next_states, reward, done, info = env.step(action)
        episode_reward += reward
        states = next_states
        if done:
            states = env.reset()
            print('Episode reward', episode_reward)
            episode_reward = 0
    
    env.close()


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    replay_buffer = ReplayBuffer(params['buffer_size'])
    exploration = LinearSchedule(schedule_timesteps=int(params['exploration_fraction'] * params['total_timesteps']),
                                 initial_p=1.0,
                                 final_p=params['exploration_final_eps'])

    with tf.Session() as sess:
        agent = DQN(env, params)
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=10)
        sess.run(agent.update_target_params)

        episode_rewards = [0.0]
        states = env.reset()
        loss = 100.0
        reset = True
        for t in range(params['total_timesteps']):
            update_eps = exploration.value(t)
            
            action = agent.get_action(sess, states, update_eps)
            next_states, reward, done, _ = env.step(action)
            replay_buffer.add(states, action, reward, next_states, float(done))
            states = next_states
            
            episode_rewards[-1] += reward
            if done:
                states = env.reset()
                episode_rewards.append(0.0)
                
            if t > learning_starts and t % train_freq == 0:
                states_t, actions, rewards, states_t1, dones = replay_buffer.sample(params['batch_size'])
                experience = [states_t, actions, rewards, states_t1, dones]
                loss = agent.train_step(sess, experience)

            if t > learning_starts and t % target_network_update_freq == 0:
                sess.run(agent.update_target_params)

            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            num_episodes = len(episode_rewards)
            if done and len(episode_rewards) % print_freq == 0:
                print('steps: %d, episodes: %d, mean_100ep_reward: %d, loss: %.4f, epsion: %.4f' %
                      (t, num_episodes, mean_100ep_reward, loss, exploration.value(t)))

        print('steps: %d, episodes: %d, mean_100ep_reward: %d, loss: %.4f, epsion: %.4f' %
              (t, num_episodes, mean_100ep_reward, loss, exploration.value(t)))

        enjoy(sess, agent, env)
        
