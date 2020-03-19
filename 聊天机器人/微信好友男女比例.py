# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 18:24:24 2019

@author: Maibenben
"""

from wxpy import *
bot = Bot()


from wxpy import *
from pyecharts import pie
import webbrowser
bot=Bot(cache_path=True) #注意手机确认登录

friends=bot.friends()
#拿到所有朋友对象，放到列表里
attr=['男朋友','女朋友','未知性别']
value=[0,0,0]
for friend in friends:
    if friend.sex == 1: # 等于1代表男性
        value[0]+=1
    elif friend.sex == 2: #等于2代表女性
        value[1]+=1
    else:
        value[2]+=1


pie = Pie("朋友男女比例")
pie.add("", attr, value, is_label_show=True)
#图表名称str，属性名称list，属性所对应的值list，is_label_show是否现在标签
pie.render('sex.html')#生成html页面
# 打开浏览器
webbrowser.open("sex.html")













