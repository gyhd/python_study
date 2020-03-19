# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 19:10:41 2019

@author: Maibenben
"""

import json
import requests
from wxpy import *
bot = Bot(cache_path=False)

group=bot.groups().search('群名字')[0]
print(group)

# 调用图灵机器人API，发送消息并获得机器人的回复
def auto_reply(text):
    url = "http://www.tuling123.com/openapi/api"
    api_key = "9d602fe417464cd18beb2083d064bee6"
    payload = {
        "key": api_key,
        "info": text,
    }
    r = requests.post(url, data=json.dumps(payload))
    result = json.loads(r.content)
    return "[来自智能机器人] " + result["text"]


@bot.register(chats=group)
def forward_message(msg):
    return auto_reply(msg.text)

embed()


