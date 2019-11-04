import gym
import keras
import datetime as dt
import tensorflow as tf
import random
import numpy as np
import math
from tensorflow.keras.layers import Dense
from tqdm import tqdm
from gym import wrappers

STORE_PATH = '/run'
MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.0005
GAMMA = 0.95
BATCH_SIZE = 40
TAU = 0.08
RANDOM_REWARD_STD = 1.0
# env = gym.make("MountainCar-v0")
state_size = 4
env = gym.make("CartPole-v1")

num_actions = env.action_space.n



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


class QN(tf.keras.Model):
    def __init__(self):
        super(QN,self).__init__()
        self.l1 = Dense(30, input_shape=(4,), kernel_initializer='random_uniform',
                bias_initializer='random_uniform')
        self.l2 = Dense(35, kernel_initializer='random_uniform',
                bias_initializer='random_uniform')

        # self.l21 = Dense(90, kernel_initializer='random_uniform',
        #         bias_initializer='random_uniform')
        self.l3 = Dense(num_actions, kernel_initializer='random_uniform',
                bias_initializer='random_uniform')

    def call(self, input):
        feat = tf.nn.relu(self.l1(input))
        feat = tf.nn.relu(self.l2(feat))
        # feat = tf.nn.relu(self.l21(feat))
        value = self.l3(feat)
        return value

def choose_action(state, primary_network, eps):
    if random.random() < eps:
        return random.randint(0, num_actions - 1)
    else:
        state = np.expand_dims(np.array(state),axis=0) #otherwise throuhg eerror..
        return np.argmax(primary_network(state))


def train(primary_network, memory, tarket_network):
    if memory.num_samples < BATCH_SIZE*3:
        return 0
    else:
        batch = memory.sample(BATCH_SIZE)
        states = np.array([val[0] for val in batch])
        actions = np.array([val[1] for val in batch])
        rewards = np.array([val[2] for val in batch])
        next_states = np.array([(np.zeros(state_size)
                                 if val[3] is None else val[3]) for val in batch])

        prim_qt = primary_network(np.expand_dims(states,axis=0)) # Q_t[s,a]
        prim_qtp1 = primary_network(np.expand_dims(next_states,axis=0)) #Q_{t+1}[s_{t+1},a_{t+1}]

        updates = rewards
        valid_idxs = np.array(next_states).sum(axis=1) != 0
        batch_idxs = np.arange(BATCH_SIZE)

        opt_q_tp1_eachS = np.argmax(np.squeeze(prim_qtp1.numpy()), axis=1) # Argmax a_{t+1} Q_{t+1} [ s_{t+1}, a_{t+1}]
        q_from_target = target_network(np.expand_dims(next_states, axis=0)) #Q^{target} [ s, a]

        updates[valid_idxs] += GAMMA*np.squeeze(q_from_target.numpy())[valid_idxs, opt_q_tp1_eachS[valid_idxs]] # update = r + \gamma Q[s_{t+1}, a^{*}_{t+1}]; with a^{*}_{t+1} = ArgMax Q_{s_t+1, a_t+1}

        ###### In the disc case... a_t = \beta_1 .... > Q[\beta_1] -> Q^{target}[n_1, \beta_1; \beta_2^{*}] with \beta_2^{*} = ArgMax Q[n_1, \beta1, \BB2]
        #consequences: for each state in the first layer, the action will be someone. 

        target_q = np.squeeze(prim_qt.numpy())
        target_q[batch_idxs, actions] = updates

        with tf.device("/cpu:0"):
            with tf.GradientTape() as tape:
                tape.watch(primary_network.trainable_variables)
                predicted_q = primary_network(states)
                target_q = np.expand_dims(target_q,axis=0)
                loss = tf.keras.losses.MSE(predicted_q, target_q)
                loss = tf.reduce_mean(loss)
                grads = tape.gradient(loss, primary_network.trainable_variables)
                optimizer.apply_gradients(zip(grads, primary_network.trainable_variables))

        for t, e in zip(target_network.trainable_variables, primary_network.trainable_variables):
            t.assign(t*(1-TAU) + e*TAU)

        return loss



save=True
#

for agent in range(1):
    env = gym.make("CartPole-v0")
    env = wrappers.Monitor(env, './videos/' + str(2) + '/')
    primary_network = QN()
    target_network = QN()
    optimizer = tf.keras.optimizers.Adam(lr=0.01)

    num_episodes  = 200
    eps = 1
    render = False
    # train_writer = tf.summary.create_file_writer("summarie/1")
    steps = 0
    rews=[]
    times=[]


    for i in range(num_episodes):

        state = env.reset()
        cnt=0
        avg_loss=0
        while True:
            # env.render()
            action = choose_action(state, primary_network, eps)
            next_state, reward, done, info = env.step(action)
            reward = np.random.normal(1.0, RANDOM_REWARD_STD)
            if cnt==300:
                done = True
            if done:
                next_state = None
            memory.add_sample((state, action, reward, next_state))
            loss = train(primary_network, memory, target_network)
            avg_loss += loss
            state = next_state
            steps +=1

            eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON)*np.exp(- LAMBDA*steps)
            if done:
                avg_loss /= cnt
                print(f"Episode: {i}, Reward: {cnt}, avg loss: {avg_loss:.3f}, eps: {eps:.3f}")
                rews.append(cnt)
                times.append(i+1)
                # with train_writer.as_default():
                #     tf.summary.scalar('reward', cnt, step=i)
                #     tf.summary.scalar('avg loss', avg_loss, step=i)
                break
            cnt += 1
    if save:
        np.save("data"+str(agent),np.array([times, rews]), allow_pickle=True)
