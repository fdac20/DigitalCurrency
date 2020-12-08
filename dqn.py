
import pandas as pd
import os
import random
import numpy as np
import time
from keras.layers import Dense, Lambda, Layer, Input
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.models import load_model
from keras import backend as K
#from keras.utils import plot_model



class DQNAgent():
    def __init__(self,parameter):
        self.state_size = parameter[2]["state_size"]
        self.action_size = parameter[2]["action_size"]
        self.memory_size = parameter[2]["memory_size"]
        self.memory = [None]*parameter[2]["memory_size"]
        self.gamma = parameter[2]["gamma"]
        self.epsilon = 1.0
        self.epsilon_min =  parameter[2]["epsilon"]
        self.train_interval = 10
        # linear decrease rate
        self.epsilon_decrement = (self.epsilon - parameter[2]["epsilon"])* self.train_interval/ (parameter[2]["iteration"]*parameter[1]["episode_length"])
        self.learning_rate = parameter[2]["learning_rate"]
        self.Update_target_frequency = 100
        self.batch_size = parameter[2]["batch_size"]
        self.model = self.build_model()
        self.model_ = self.build_model()
        self.i = 0
        self.train_test = parameter[0]["mode"]
        self.symbol=''
    def save_model(self):
        self.model.save(r'./Saved Models/'+self.symbol+'.h5')
    def load_model(self):
        print(self.symbol, "path=",'./Saved Models/'+self.symbol+'.h5')
        self.model = load_model(r'./Saved Models/'+self.symbol+'.h5')
    #build deep neural network
    def build_model(self):
        model = Sequential()
        neurons_per_layer = 24
        activation = "relu"
        model.add(Dense(neurons_per_layer,
                        input_dim=self.state_size,
                        activation=activation))
        model.add(Dense(neurons_per_layer * 2, activation=activation))
        model.add(Dense(neurons_per_layer * 4, activation=activation))
        model.add(Dense(neurons_per_layer * 2, activation=activation))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        #plot_model(model,show_shapes=True,to_file='model.png')
        return model
   #Acting Policy of the DQNAgent
    def act(self, state, test=False):
        act_values=[]
        action = np.zeros(self.action_size)
        #epsilon explore policy
        if np.random.rand() <= self.epsilon and self.train_test == 'train' and not test:
            action[random.randrange(
                self.action_size)] = 1  # saeed : it would put 1 in either of the 3 action positions randomly
        else:
        #greedy  behavoir policy
            state = state.reshape(1, self.state_size)
            act_values = self.model.predict(state)
            action[np.argmax(act_values[0])] = 1
        if test:
            return action, act_values
        else:
            return action
    #Memory Management and training of the agent for 1 epoch
    def observe(self, state, action, reward, next_state, done, warming_up=False):
        self.i = (self.i + 1) % self.memory_size
        self.memory[self.i] = (state, action, reward, next_state, done)
        if (self.i == self.memory_size - 1):
            pass
        if (not warming_up) and (self.i % self.train_interval) == 0:
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decrement
            state, action, reward, next_state, done = self._get_batches()
            reward += (self.gamma* np.logical_not(done)* np.amax(self.model.predict(next_state),axis=1))
            q_target = self.model.predict(state)
            q_target[action[0], action[1]] = reward
            return self.model.fit(state, q_target,batch_size=self.batch_size,epochs=1,verbose=False,validation_split=0.1)
    #Selecting a batch of memory and split it into categorical subbatches. Process action_batch into a position vector
    def _get_batches(self):
        batch = np.array(random.sample(self.memory, self.batch_size))
        state_batch = np.concatenate(batch[:, 0]).reshape(self.batch_size, self.state_size)
        action_batch = np.concatenate(batch[:, 1]).reshape(self.batch_size, self.action_size)
        reward_batch = batch[:, 2]
        next_state_batch = np.concatenate(batch[:, 3]).reshape(self.batch_size, self.state_size)
        done_batch = batch[:, 4]
        # action processing
        action_batch = np.where(action_batch == 1)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
