# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 21:43:22 2019

@author: Maibenben
"""


import itchat
import time

@itchat.msg_register(itchat.content.TEXT)
def reply_msg(msg):
    print("收到一条信息：",msg.text)

if __name__ == '__main__':
    itchat.auto_login()
    time.sleep(5)
    itchat.send("文件助手你好哦", toUserName="File Transfer")
    itchat.run()




"""

import itchat
import time


def after_login():
    print("登录后调用")


def after_logout():
    print("退出后调用")


if __name__ == '__main__':
    itchat.auto_login(loginCallback=after_login, exitCallback=after_logout)
    time.sleep(50000)
    itchat.logout()




import itchat
import time

def after():
    user_info = itchat.search_friends(name='姜汁儿')
    if len(user_info) > 0:
        # 拿到用户名
        user_name = user_info[0]['UserName']
        # 发送文字信息
        itchat.send_msg('你好啊！', user_name)
        # 发送图片
        #time.sleep(10)
        #itchat.send_image('cat.jpg', user_name)
        # 发送文件
        #time.sleep(10)
        #itchat.send_file('19_2.py', user_name)
        # 发送视频
        #time.sleep(10)
        #itchat.send_video('sport.mp4', user_name)


if __name__ == '__main__':
    itchat.auto_login(loginCallback=after)
    itchat.run()


"""












