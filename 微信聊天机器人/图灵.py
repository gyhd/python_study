# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:00:09 2019
@author: Maibenben
"""


#要可以登录微信网页版的才可以用
import json
import requests
from wxpy import *


def reply(text):
    url = "http://www.tuling123.com/openapi/api"
    api_key = "f3290495fac64618ba333f159ef01d22"
    payload = {
            "key" : api_key,
            "api_key" : text,
            "userId" : "nibaba"
            }
    r = requests.post(url,data = json.dumps(payload))
    result = json.load(r.content)
    if ('url' in result.key()):
        return ""+result["text"]+result["url"]
    else:
        return ""+result["text"]
    
bot = Bot(cache_path = "botoo.pkl")#登录缓存
print("nibaba on line")
found = bot.friends().search("郭威伟")
print(found)


@bot.register(found)
def message(msg):
    ret = reply(msg.text)
    return ret

@bot.register(found)
def forward_message(msg):
    ret = reply(msg.text)
    return ret

embed()



