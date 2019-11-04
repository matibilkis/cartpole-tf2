import gym
import keras
import datetime as dt
import tensorflow as tf
import random
import numpy as np
import math


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


primary_network = keras.Sequential()
primary_network.add(keras.layers.Dense(30, input_dim=4,activation='relu', kernel_initializer=keras.initializers.he_normal()))
primary_network.add(keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()))
primary_network.add(keras.layers.Dense(num_actions))

primary_network.compile(optimizer=keras.optimizers.Adam(), loss='mse')

import numpy as np
data = np.random.random((1,4))

primary_network( data )
