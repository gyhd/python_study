# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 19:09:31 2019

@author: Maibenben
"""

import json
import requests
from wxpy import *
bot = Bot(cache_path=True)

# 调用图灵机器人API，发送消息并获得机器人的回复
def auto_reply(text):
    url = "http://www.tuling123.com/openapi/api"
    api_key = "9df516a74fc443769b233b01e8536a42"
    payload = {
        "key": api_key,
        "info": text,
    }
    r = requests.post(url, data=json.dumps(payload))
    result = json.loads(r.content)
    return "[来自智能机器人] " + result["text"]


@bot.register()
def forward_message(msg):
    return auto_reply(msg.text)

embed()


