# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:12:37 2019

@author: Maibenben
"""

from wxpy import *

bot = Bot()
my_friend = bot.friends().search('程', sex=FEMALE, city='zhengzhou')[0]
# <Friend: 游否>

# 发送文本
my_friend.send('Hello, WeChat!')
# 发送图片
#my_friend.send_image('my_picture.png')
# 发送视频
#my_friend.send_video('my_video.mov')
# 发送文件
#my_friend.send_file('my_file.zip')
# 以动态的方式发送图片
#my_friend.send('@img@my_picture.png')


