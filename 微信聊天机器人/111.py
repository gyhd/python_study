# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:03:50 2019

@author: Maibenben
"""

from wxpy import *
bot = Bot()

# 机器人账号自身
myself = bot.self

# 向文件传输助手发送消息
bot.file_helper.send('Hello from wxpy!')

