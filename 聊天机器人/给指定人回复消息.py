# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 19:12:21 2019

@author: Maibenben
"""

import requests
from wxpy import *
bot = Bot( cache_path=True)

girl_friend=bot.search('名字r')[0]

# 调用图灵机器人API，发送消息并获得机器人的回复
def auto_reply(text):
    url = "http://www.tuling123.com/openapi/api"
    api_key = "申请图灵机器人获取key值放到这里"
    payload = {
        "key": api_key,
        "info": text,
    }
    r = requests.post(url, data=json.dumps(payload))
    result = json.loads(r.content)
    return "[微信测试，请忽略] " + result["text"]


@bot.register()
def forward_message(msg):
    if msg.sender == girl_friend:
        return auto_reply(msg.text)

embed()


