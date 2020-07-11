# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:23:50 2019

@author: Maibenben
"""



import gym
import numpy as np
import random
from IPython.display import clear_output

#初始出租车V2环境
env = gym.make("Taxi-v3").env

#初始任意值
q_table = np.zeros([env.observation_space.n, env.action_space.n])

#超参数alpha，gamma，epsilon
alpha = 0.1
gamma = 0.6
epsilon = 0.1


all_epochs = []
all_penalties = []

for i in range(1, 10001):
    state = env.reset()

    #初始化变量
    epochs, penalties, reward, = 0, 0, 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            #检查操作空间
            action = env.action_space.sample()
        else:
            #检查学习值
            action = np.argmax(q_table[state])

        next_state, reward, done, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        #更新新的值
        new_value = (1 - alpha) * old_value + alpha * \
            (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1

    if i % 100 == 0:
        clear_output(wait=True)
        print("Episode: ", i)

print("new_value = ", new_value)
print("Training finished.")













