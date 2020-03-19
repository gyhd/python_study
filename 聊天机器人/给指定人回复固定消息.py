# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 19:06:28 2019

@author: Maibenben
"""

from wxpy import *
bot=Bot(cache_path=True)

girl_friend=bot.search('刘刘刘')[0]
print(girl_friend)

@bot.register() # 接收从指定好友发来的消息，发送者即recv_msg.sender为指定好友girl_friend
def recv_send_msg(recv_msg):
    print('收到的消息：',recv_msg.text) # recv_msg.text取得文本
    if recv_msg.sender == girl_friend:
        recv_msg.forward(bot.file_helper,prefix='老婆留言: ') #在文件传输助手里留一份，方便自己忙完了回头查看
        ms='老婆最美丽，我对老婆的爱如滔滔江水，连绵不绝'
        print('>>>给老婆回复的：', ms)
        return  ms#给老婆回一份

embed()





