import numpy as np
import tensorflow as tf


class Reinforce(object):
    def __init__(self, env):
        self.env = env

        self.value_size = 1
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.shape[0]
        self.input_shape = (None, self.state_size)

        self.discount_factor = 0.99
        self.lr = 0.001

        self.add_ops()
        self.policy()
        self.train_op()

    def add_ops(self):
        self.state = tf.placeholder(tf.float32, self.input_shape, name='input_shape')
        self.action = tf.placeholder(tf.int32, [None], name='policy_action')
        self.reward = tf.placeholder(tf.float32, [None, 1], name='expected_reward')

    def policy(self):
        with tf.variable_scope('policy'):
            h = tf.layers.dense(self.state, 32, activation=tf.nn.tanh, use_bias=True)
            h = tf.layers.dense(h, 16, activation=tf.nn.tanh, use_bias=True)
            self.p = tf.layers.dense(h, self.action_size, activation=tf.nn.softmax, use_bias=True)

            action = tf.one_hot(self.action, self.action_size)
            J = tf.reduce_sum(tf.log(self.p + 1e-8) * action, axis=-1) * self.reward
            self.loss = -tf.reduce_mean(J)

    def train_op(self):
        tvars = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(self.lr)
        grads_and_vars_actor = optimizer.compute_gradients(self.loss, var_list=tvars)
        self.train_op = optimizer.apply_gradients(grads_and_vars_actor)

    def train_step(self, sess, batch):
        # batch = [state, action, reward]
        state, action, reward = batch

        state = np.reshape(state, (-1, self.state_size))
        action = np.reshape(np.array(action), (-1, ))
        reward = np.reshape(np.array(reward), (-1, 1))

        feed_dict = {
            self.state: state,
            self.action: action,
            self.reward: reward
        }

        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        
        return loss

    def get_action(self, sess, state, istrain=True):
        state = np.reshape(state, (-1, self.state_size))
        feed_dict = {self.state: state}
        policy = sess.run(self.p, feed_dict)

        if istrain:
            return np.random.choice(self.action_size, 1, p=policy[0])[0]
        else:
            return np.argmax(policy[0])
