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



class QN(tf.keras.Model):
    def __init__(self):
        super(QN,self).__init__()
        self.l1 = Dense(30, input_shape=(4,), kernel_initializer='random_uniform',
                bias_initializer='random_uniform')
        self.l2 = Dense(30, kernel_initializer='random_uniform',
                bias_initializer='random_uniform')
        self.l3 = Dense(num_actions, kernel_initializer='random_uniform',
                bias_initializer='random_uniform')

    def call(self, input):
        feat = tf.nn.relu(self.l1(input))
        feat = tf.nn.relu(self.l2(feat))
        value = self.l3(feat)
        return value

primary_network = QN()
target_network = QN()

# optimizer = tf.keras.optimizers.Adam(lr=0.01, loss="mse")


class Memory():
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._samples = []
    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)
    def sample(self, no_samples):
        if no_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        else:
            return random.sample(self._samples, no_samples)
    @property
    def num_samples(self):
        return len(self._samples)
memory = Memory(50000)



def choose_action(state, primary_network, eps):
    if random.random() < eps:
        return random.randint(0, num_actions - 1)
    else:
        state = np.expand_dims(np.array(state),axis=0) #otherwise throuhg eerror..
        return np.argmax(primary_network(state))



def train(primary_network, memory, target_network=None):
    if memory.num_samples < BATCH_SIZE * 3:
        return 0
    batch = memory.sample(BATCH_SIZE)
    states = np.array([val[0] for val in batch])
    actions = np.array([val[1] for val in batch])
    rewards = np.array([val[2] for val in batch])
    next_states = np.array([(np.zeros(state_size)
                             if val[3] is None else val[3]) for val in batch])
    # predict Q(s,a) given the batch of states
    prim_qt = primary_network(np.expand_dims(states,axis=0))
    # predict Q(s',a') from the evaluation network
    prim_qtp1 = primary_network(np.expand_dims(next_states,axis=0))
    # copy the prim_qt into the target_q tensor - we then will update one index corresponding to the max action
    target_q = prim_qt.numpy()
    updates = rewards
    valid_idxs = np.array(next_states).sum(axis=1) != 0
    batch_idxs = np.arange(BATCH_SIZE)
    if target_network is None:
        print(prim_qtp1.numpy())
        print(***************)
        updates[valid_idxs] += GAMMA * np.amax(prim_qtp1.numpy()[valid_idxs, :], axis=1)
    else:
        prim_action_tp1 = np.argmax(prim_qtp1.numpy(), axis=1)
        q_from_target = target_network(np.expand_dims(next_states,axis=0))
        updates[valid_idxs] += GAMMA * q_from_target.numpy()[batch_idxs[valid_idxs], prim_action_tp1[valid_idxs]]
    target_q[batch_idxs, actions] = updates

    # with tf.device("/cpu:0"):
    #     with tf.GradientTape() as tape:
    #         tape.watch(primary_network.trainable_variables)
    #         loss = tf.keras.losses(states, )
    loss = primary_network.train_on_batch(np.expand_dims(states,axis=0), target_q)
    if target_network is not None:
        # update target network parameters slowly from primary network
        for t, e in zip(target_network.trainable_variables, primary_network.trainable_variables):
            t.assign(t * (1 - TAU) + e * TAU)
    return loss




num_episodes = 1000
eps = MAX_EPSILON
render = False
train_writer = tf.summary.create_file_writer(STORE_PATH + f"/DoubleQ_{dt.datetime.now().strftime('%d%m%Y%H%M')}")
double_q = False
steps = 0
for i in range(num_episodes):
    state = env.reset()
    cnt = 0
    avg_loss = 0
    while True:
        if render:
            env.render()
        action = choose_action(state, primary_network, eps)
        next_state, reward, done, info = env.step(action)
        reward = np.random.normal(1.0, RANDOM_REWARD_STD) #to add stochasisticity
        if done:
            next_state = None
        # store in memory
        memory.add_sample((state, action, reward, next_state))
        loss = train(primary_network, memory, target_network if double_q else None)
        avg_loss += loss
        state = next_state
        # exponentially decay the eps value
        steps += 1
        eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * steps)
        if done:
            avg_loss /= cnt
            print(f"Episode: {i}, Reward: {cnt}, avg loss: {avg_loss:.3f}, eps: {eps:.3f}")
            with train_writer.as_default():
                tf.summary.scalar('reward', cnt, step=i)
                tf.summary.scalar('avg loss', avg_loss, step=i)
            break
        cnt += 1






#
