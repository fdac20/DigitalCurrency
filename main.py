import numpy as np
import pandas as pd
import random
import re
import os
from Custom_data_preprocessor import data_processor
from enviroment import trade_enviroment
from dqn import DQNAgent


#define and intialize parameters
data_parameters = {"dataname": r'./Data/daily_price_eth_cny.csv',
                "mode":'train',
                "split":0.75,
                 }

environment_parameters = {"trading_fee":.0001,
                                "time_fee":.001,
                                "episode_length":0,
                                }
training_parameters = {"memory_size":3000,
                       "iteration":10,
                       "batch_size":64,
                       "learning_rate":0.001,
                       "gamma":0.96,
                       "epsilon":0.01,
                       "state_size":0,
                       "action_size":0,
                            }
parameter = [data_parameters, environment_parameters, training_parameters]


#preprocess data
def data_prepare(parameter):
    data = data_processor(filename=parameter[0]["dataname"],mode=parameter[0]["mode"],split=parameter[0]["split"])
    episode_length = round(int(len(pd.read_csv(parameter[0]["dataname"]))*parameter[0]["split"]), -1)
    return data, episode_length

def create_states(parameter, data):
    environment = trade_enviroment(data,parameter)
    state = environment.reset()
    return len(environment._actions),len(state),state,environment

def train(agent,parameter,state,environment):
    # Training the agent
    for m in range(parameter[2]["memory_size"]):
        action = agent.act(state)
        next_state, reward, done, _ = environment.step(action)
        agent.observe(state, action, reward, next_state, done, warming_up=True)

    loss_list=[]
    val_loss_list=[]
    reward_list=[]
    epsilon_list=[]
    metrics_df=None
    best_loss = 9999
    best_reward = 0

    for i in range(parameter[2]["iteration"]):
        state_space = environment.reset()
        rew = 0
        loss_list_temp = []
        val_loss_list_temp = []
        for j in range(parameter[1]["episode_length"]):
            action = agent.act(state_space)
            next_state, reward, done, _ = environment.step(action)
            loss = agent.observe(state_space, action, reward, next_state,done)  # loss would be none if the episode length is not % by 10
            state_space = next_state
            rew += reward
            if(loss):
                loss_list_temp.append(round(loss.history["loss"][0],3))
                val_loss_list_temp.append(round(loss.history["val_loss"][0],3))
        print("iteration:" + str(i)+ "| rewward:" + str(round(rew, 2))+ "| eps:" + str(round(agent.epsilon, 2))+ "| loss:" + str(round(loss.history["loss"][0], 4)))
        print("Loss=", str(np.mean(loss_list_temp)), " Val_Loss=", str(np.mean(val_loss_list_temp)))
        loss_list.append(np.mean(loss_list_temp))
        val_loss_list.append(np.mean(val_loss_list_temp))
        reward_list.append(rew)
        epsilon_list.append(round(agent.epsilon, 2))

    #save model
    agent.save_model()
    metrics_df=pd.DataFrame({'loss':loss_list,'val_loss':val_loss_list,'reward':reward_list,'epsilon':epsilon_list})
    metrics_df.to_csv(r'./Results/perf_metrics.csv')

#calculate Q_talbe, total reward, policy
def caculation(agent, env):

    state = env.reset()
    q_values_list=[]
    state_list=[]
    action_list=[]
    reward_list=[]

    i = False
    while not i:
        action, q_values = agent.act(state, test=True)
        state_space, reward, i, info = env.step(action)
        if 'status' in info and info['status'] == 'Closed plot':
            i = True
        else:
            reward_list.append(reward)
        q_values_list.append(q_values)
        state_list.append(state)
        action_list.append(action)

    print('Reward = %.2f' % sum(reward_list))
    action_policy_df = pd.DataFrame({'q_values':q_values_list,'state':state_list,'action':action_list})
    if parameter[0]["mode"] == 'train':
        action_policy_df.to_pickle(r'./Results/action_policy_train.pkl')
    elif parameter[0]["mode"] == 'test':
        action_policy_df.to_pickle(r'./Results/action_policy_test.pkl')



def main():
    train_data, parameter[1]["episode_length"] = data_prepare(parameter)
    parameter[2]['action_size'], parameter[2]['state_size'], state, env = create_states(parameter, train_data)
    #create model
    agent = DQNAgent(parameter)
    #train model and save
    train(agent,parameter,state,env)
    caculation(agent,env)

    #test model
    parameter[0]["mode"] = 'test'
    test_data = data_prepare(parameter)[0]
    test_env = create_states(parameter, test_data)[3]
    caculation(agent,test_env)

main()
