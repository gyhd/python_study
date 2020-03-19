# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 22:54:49 2019

@author: Maibenben
"""

from wxpy import *

bot = Bot(cache_path=True)
#my_frends = bot.friends().search(u'王林')[0]
#my_frends.send('11')                  #给朋友发消息
bot.file_helper.send('Hello World!')  #给文件助手发消息
#bot.self.send('Hello World!')   #给机器人自己发消息

#print(my_frends)









