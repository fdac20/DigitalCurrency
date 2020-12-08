import pickle
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import re

actions_dict = {
        'hold': (1, 0, 0),
        'buy': (0, 1, 0),
        'sell':(0, 0, 1),
    }

#Reverse the dictionary of action
def action_trans():
    action_dict_v = {v:k for k,v in actions_dict.items()}
    return action_dict_v
#read pd data
def pd_to_list(data,action_dict_v):
    action_price_pairs = []
    data = np.array(data)
    for x in data:
        action_price_pairs.append(action_dict_v[tuple(list(x[0].astype(int)))])

    return action_price_pairs
#get the price history
def get_price(data_price, index, mode):
    data_price_return = []
    if mode =="train":
        data_train = data_price[0:index]
        for x in data_train:
            data_price_return.append(x[2])
    elif mode =="test":
        data_test = data_price[-(index+1):-1]
        print(data_test)
        for x in data_test:
            data_price_return.append(x[2])
    return data_price_return
#get the policy
with open(r'./Results/eth_test/action_policy.pkl', 'rb') as f:
    data_policy = pickle.load(f)

data_price = pd.read_csv(r'./Data/eth.csv').values.tolist()
index = np.array(data_policy).shape[0]


action_dict_v = action_trans()
a = pd_to_list(data_policy,action_dict_v)
b = get_price(data_price,index,"test" )
print(len(a),len(b))
#calcualte the profit according to policy
def profit_cal(price, action):
    number = 800
    invest = 100000
    final = 0
    c = invest+number*price[0]
    profit_total = []
    print('initial number:',number,'initial money:',invest)
    for i in range(1,len(action)):

        if action[i] == 'buy':
            if invest == 0 and invest < price[i]:
                number = number
                invest = invest
            else:
                number += 1
                invest += -1*price[i]
        elif action[i] =='hold':
            number = number
            invest = invest
        elif action[i] == 'sell':
            if number == 0:
                number = number
                invest = invest
            else:
                number += -1
                invest += 1*price[i]
        print(number,invest,price[i],action[i])
        final = price[i]*number+invest
        profit_total.append(((final-c)/c)*100)
    return profit_total, number, invest


r, n, i = profit_cal(b,a)
print('final number:', n, 'final money:' ,i)
def to_precent(temp, position):
    return '%1.0f'%temp+'%'
#plot results
plt.plot(r)
my_y_ticks=range(-100,300,50)
plt.grid()
plt.yticks(my_y_ticks)
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_precent))
plt.ylabel("Percentage")
plt.xlabel("hours")
plt.title("Profit of trading ltc short term with high frequency(test)")

plt.show()
