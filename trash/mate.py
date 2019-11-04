import gym
import keras
import datetime as dt
import tensorflow as tf
import random
import numpy as np
import math
from tensorflow.keras.layers import Dense

STORE_PATH = '/run'
MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.0005
GAMMA = 0.95
BATCH_SIZE = 32
TAU = 0.08
RANDOM_REWARD_STD = 1.0
env = gym.make("CartPole-v1")
# env = gym.make("MountainCar-v0")
state_size = 4
num_actions = env.action_space.n

from tensorflow.keras.layers import Dense


class QN(tf.keras.Model):
    def __init__(self, layer=0):
        super(QN,self).__init__()
        self.l1 = Dense(40, input_shape=(4,), kernel_initializer='random_uniform',
                bias_initializer='random_uniform')
        self.l2 = Dense(10, kernel_initializer='random_uniform',
                bias_initializer='random_uniform')
        self.l3 = Dense(num_actions, kernel_initializer='random_uniform',
                bias_initializer='random_uniform')

    def call(self, input):
        # state, action = input
        # features = tf.concat([state, action], axis=1)
        feat = tf.nn.relu(self.l1(input))
        feat = tf.nn.relu(self.l2(feat))
        value = self.l3(feat)
        return tf.nn.tanh(value)*2 #Tanh image is [-1,1]


f = QN()
env.reset()
action=env.action_space.sample()
state, reward, _, _ = env.step(action)
print(state)
# f(tf.constant(np.array(state)))
f( [[1., 3., 4., 5.]])
# input = np.expand_dims(np.array([state, np.array(action)]), axis=1)
# f([ [ [1.,3.,4.,5.] ] , [ [ 4. ] ] ])












#
