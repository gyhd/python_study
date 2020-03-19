# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 22:09:04 2019

@author: Maibenben
"""


image = "F:\\python\\yangyang"

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



