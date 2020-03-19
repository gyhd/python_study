# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 22:06:53 2019

@author: Maibenben
"""

from aip import AipFace

APP_ID = '16803437'
API_KEY = 'kiBnBvC4vqZVsCMUzxyxjGVT'
SECRET_KEY = 'aDqFc08cFG8lmFAzSryjeVHqGa5Z1U9C'

client = AipFace(APP_ID, API_KEY, SECRET_KEY)


image = "F:\\python\\huge\\huge-1.jpg"

imageType = "BASE64"
groupId = "group1"
userId = "user1"

""" 调用人脸注册 """
client.addUser(image, imageType, groupId, userId);

""" 如果有可选参数 """
options = {}
options["user_info"] = "user's info"
options["quality_control"] = "NORMAL"
options["liveness_control"] = "LOW"
options["action_type"] = "REPLACE"

""" 带参数调用人脸注册 """
client.addUser(image, imageType, groupId, userId, options)

