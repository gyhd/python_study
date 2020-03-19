# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 19:05:35 2019

@author: Maibenben
"""

from wxpy import *
bot=Bot(cache_path=True)

@bot.register()
def recv_send_msg(recv_msg):
    print('收到的消息：',recv_msg.text) # recv_msg.text取得文本
    return '自动回复：%s' %recv_msg.text

# 进入Python命令行，让程序保持运行
embed()







