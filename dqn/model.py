import numpy as np
import tensorflow as tf


class DQN(object):
    def __init__(self, env, args):
        self.env = env
        self.args = args

        self.hidden_size = self.args.get('hidden_size', 32)
        self.num_actions = self.env.action_space.n
        self.num_states = self.env.observation_space.shape[0]

        self._add_ops()
        self.policy()
        self.build_train()
        self.update_target()
        self.build_train_op()

    def _add_ops(self):
        self.state = tf.placeholder(tf.float32, shape=(None, self.num_states), name='state')
        self.action = tf.placeholder(tf.int32, [None], name='action')
        self.reward = tf.placeholder(tf.float32, [None], name='reward')
        self.next_state = tf.placeholder(tf.float32, shape=(None, self.num_states), name='next_state')
        self.done_mask = tf.placeholder(tf.float32, [None], name='done')

        self.stochastic = tf.placeholder(tf.bool, (), name='stochastic')
        self.update_epsion = tf.placeholder(tf.float32, (), name='update_epsion')

    def _get_qa_value(self, inputs, num_actions, scope='qa_value', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            latent = tf.layers.dense(inputs, units=64, activation=tf.tanh, name='fc1')
            # latent = tf.layers.dense(latent, units=64, activation=tf.tanh, name='fc2')
            action_scores = tf.layers.dense(latent, units=num_actions, activation=None)
        
        return action_scores

    def policy(self, scope='deepq', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            eps = tf.get_variable('eps', (), initializer=tf.constant_initializer())

            q_values = self._get_qa_value(self.state, self.num_actions, scope="q_func")
            deterministic_actions = tf.argmax(q_values, axis=1)

            batch_size = tf.shape(self.state)[0]
            random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=self.num_actions, dtype=tf.int64)
            chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
            stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

            self.output_actions = tf.cond(self.stochastic, lambda: stochastic_actions, lambda: deterministic_actions)
            self.update_eps_expr = eps.assign(tf.cond(self.update_epsion >= 0, lambda: self.update_epsion, lambda: eps))

    def build_train(self, gamma=1.0, reuse=None):
        with tf.variable_scope('deepq', reuse=reuse):
            # q network evaluation
            q_t = self._get_qa_value(self.state, self.num_actions, scope="q_func", reuse=True)
            self.q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                            scope=tf.get_variable_scope().name + "/q_func")

            # target q network evalution
            q_t1 = self._get_qa_value(self.next_state, self.num_actions, scope="target_q_func")
            self.target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                   scope=tf.get_variable_scope().name + "/target_q_func")

            q_t_selected = tf.reduce_sum(q_t * tf.one_hot(self.action, self.num_actions), 1)
            q_t1_best = tf.reduce_max(q_t1, 1)
            q_t1_best_masked = (1.0 - self.done_mask) * q_t1_best
            q_t_selected_target = self.reward + gamma * q_t1_best_masked
            
            td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)

            def huber_loss(x, delta=1.0):
                """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
                return tf.where(
                    tf.abs(x) < delta,
                    tf.square(x) * 0.5,
                    delta * (tf.abs(x) - 0.5 * delta)
                )
            errors = huber_loss(td_error)
            self.loss = tf.reduce_mean(errors)

    def update_target(self):
        update_target_expr = []
        for var, var_target in zip(sorted(self.q_func_vars, key=lambda v: v.name), 
                                   sorted(self.target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        self.update_target_params = tf.group(*update_target_expr)

    def build_train_op(self):
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(self.loss, var_list=self.q_func_vars)
        self.train_op = optimizer.apply_gradients(grads_and_vars)

    def train_step(self, sess, batch):
        feed_dict = {
            self.state: batch[0],
            self.action: batch[1],
            self.reward: batch[2],
            self.next_state: batch[3],
            self.done_mask: batch[4]
        }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        
        return loss

    def get_action(self, sess, state, rand=True, eps=-1):
        feed_dict = {
            self.state: np.reshape(state, (-1, self.num_states)),
            self.stochastic: rand,
            self.update_epsion: eps
        }
        action, _ = sess.run([self.output_actions, self.update_eps_expr], feed_dict)
        # action = sess.run(self.output_actions, feed_dict)

        return action[0]

