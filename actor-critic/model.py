import numpy as np
import tensorflow as tf


class ActorCritic(object):
    def __init__(self, env):
        self.env = env

        self.value_size = 1
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.shape[0]
        self.input_shape = (None, self.state_size)

        self.discount_factor = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.001

        self._add_ops()
        self.build_actor()
        self.build_critic()
        self.build_train()
        self.build_train_op()

    def _add_ops(self):
        self.state = tf.placeholder(tf.float32, self.input_shape, name='input_shape')
        self.action = tf.placeholder(tf.int32, [None])
        self.target = tf.placeholder(tf.float32, [None, 1])
        self.advantage = tf.placeholder(tf.float32, [None, self.action_size])
        
    def build_actor(self, scope='actor', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            latent = tf.layers.dense(self.state, units=32, activation=tf.tanh, name='fc_a')
            self.actor = tf.layers.dense(latent, units=self.action_size, activation=tf.nn.softmax, name='actor_a')
            self.actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)

    def build_critic(self, scope='critic', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            latent = tf.layers.dense(self.state, units=32, activation=tf.tanh, name='fc_c')
            self.critic = tf.layers.dense(latent, units=self.value_size, name='critic_c')
            self.critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)

    def build_train(self):
        # temp = tf.nn.softmax_cross_entropy_with_logits(logits=self.actor, labels=self.advantage)
        temp = -tf.log(self.actor + 1e-8) * self.advantage
        self.pg_loss = tf.reduce_mean(temp)
        self.vf_loss = tf.reduce_mean(tf.losses.mean_squared_error(self.target, self.critic))

    def build_train_op(self):
        optimizer = tf.train.AdamOptimizer(self.actor_lr)
        grads_and_vars_actor = optimizer.compute_gradients(self.pg_loss, var_list=self.actor_vars)
        self.train_op_actor = optimizer.apply_gradients(grads_and_vars_actor)

        optimizer_c = tf.train.AdamOptimizer(self.actor_lr)
        grads_and_vars_critic = optimizer_c.compute_gradients(self.vf_loss, var_list=self.critic_vars)
        self.train_op_critic = optimizer_c.apply_gradients(grads_and_vars_critic)

    def train_step(self, sess, batch):
        # batch = [states, actions, adv, rewards]
        state, action, reward, next_state, done = batch

        target = np.zeros((1, 1))
        advantages = np.zeros((1, self.action_size))

        value = sess.run(self.critic, feed_dict={self.state: state})[0]
        next_value = sess.run(self.critic, feed_dict={self.state: next_state})[0]

        if done:
            advantages[0][action] = reward - value
            target[0][0] = reward
        else:
            advantages[0][action] = reward + self.discount_factor * next_value - value
            target[0][0] = reward + self.discount_factor * next_value

        feed_dict = {
            self.state: state,
            self.target: target,
            self.advantage: advantages
        }

        output = {
            'pg_loss': self.pg_loss,
            'vf_loss': self.vf_loss,
            'pi': self.actor,
            'vf': self.critic
        }

        _, _, output = sess.run([self.train_op_actor, self.train_op_critic, output], feed_dict)
        
        return output

    def get_action(self, sess, state):
        state = np.reshape(state, (-1, self.state_size))
        feed_dict = {self.state: state}
        policy = sess.run(self.actor, feed_dict)

        return np.random.choice(self.action_size, 1, p=policy[0])[0]
