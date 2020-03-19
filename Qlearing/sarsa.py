# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 17:27:16 2019

@author: Maibenben
"""

import gym
import random

env = gym.make('Taxi-v2')

# 学习率
alpha = 0.5
# 折扣因子
gamma = 0.9
# ε
epsilon = 0.05

# 初始化Q表
Q = {}
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        Q[(s, a)] = 0.0


# ε-贪婪策略选取动作
def epsilon_greedy_policy(state, epsilon):
    # 如果＜ε，随机选取一个另外的动作（探索）
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    # 否则，选取令当前状态下Q值最大的动作（开发）
    else:
        return max(list(range(env.action_space.n)), key=lambda x: Q[(state, x)])


# 训练1000个episode
for i in range(1000):
    r = 0
    # 初始化状态（env.reset()用于重置环境）
    state = env.reset()
    # 采用ε-贪婪策略选取动作
    action = epsilon_greedy_policy(state, epsilon)
    # 一个episode
    while True:
        # 输出当前agent和environment的状态（可删除）
        env.render()
        # 执行动作得到的一些信息
        nextstate, reward, done, _ = env.step(action)
        # 采用ε-贪婪策略选取下一步的动作
        nextaction = epsilon_greedy_policy(nextstate, epsilon)
        # 更新Q表
        Q[(state, action)] += alpha * (reward + gamma * Q[(nextstate, nextaction)] - Q[(state, action)])
        # a ⬅ a'
        action = nextaction
        # s ⬅ s'
        state = nextstate
        # 累加奖励
        r += reward
        # 判断episode是否到达最终状态
        if done:
            break
    # 打印当前episode的奖励
    print("[Episode %d] Total reward: %d" % (i + 1, r))
env.close()



