# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 17:00:39 2019

@author: Maibenben
"""


from threading import Timer
from itchat.content import *
import requests
import itchat


#获取金山词霸每日一句英语
def getNews():
	#打开金山词霸开放平台API
	url = "http://open.iciba.com/dsapi"
	req = requests.get(url)
	#"content":"Interests are anchors, and I believe they will bring peace and even happiness in the end."
	contents = req.json()['content']
	trans = req.json()['translation']
	return contents,trans


#发送消息
def sendNews():
	#global my_love
	try:
		#会弹出网页二维码扫描登录微信
		itchat.auto_login() 
		#不想每次都扫描，登录时预配置
		#itchat.auto_login(hotReload=True)
		#itchat.run()
		#1.想给谁发信息，先找到该朋友，备注名
		my_friend = itchat.search_friends(name = r'陈某')
		#2.找到UserName
		my_love = my_friend[0]["UserName"]
		#new_cont, new_trans = getNews()
		#print(new_cont)
		#print(new_trans)
		msg1 = str(getNews()[0]) #获取金山词霸字典内容
		msg2 = str(getNews()[1][5:])
		msg = '翻译：'
		cont = str(msg+msg2)
		#3.发送消息
		#调试，直接给微信手机助手发送消息
		itchat.send(msg1,toUserName = "filehelper")
		itchat.send(cont,toUserName = 'filehelper')
		#给微信好友发送消息
		itchat.send(msg1,toUserName = my_love)
		itchat.send(cont,toUserName = my_love)
		#每隔86400秒发送一次，每天发送一次
		Timer(8,sendNews).start()
	except:
		msg4 = "最爱你的人来啦！！！"
		itchat.send(msg4,toUserName = my_friend)
		#itchat.logout()
        
#调试函数
def test():
	itchat.auto_login()
	itchat.send(u'测试消息发送','filehelper')#发送给文件助手

if __name__ == '__main__':
	sendNews()
    







