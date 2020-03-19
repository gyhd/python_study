# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 10:53:56 2019

@author: Maibenben
"""


from aip import AipFace
import datetime
# encoding:utf-8
import base64
import urllib
import ssl


#我的第一个百度人脸检测的api
APP_ID = '16803437'
API_KEY = 'kiBnBvC4vqZVsCMUzxyxjGVT'
SECRET_KEY = 'aDqFc08cFG8lmFAzSryjeVHqGa5Z1U9C'

client = AipFace(APP_ID, API_KEY, SECRET_KEY)


##获取访问的token
def getAccessToken():
    # client_id 为官网获取的AK， client_secret 为官网获取的SK
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id='+API_KEY+'&client_secret='+SECRET_KEY
    request = urllib.Request(host)
    request.add_header('Content-Type', 'application/json; charset=UTF-8')
    response = urllib.urlopen(request)
    content = response.read()
    if (content):
        print(content)
    return content


# 读取图片
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

 # 调用人脸属性检测接口
def faceDetecting(picName,options=""):
    begin = datetime.datetime.now()
    result = ""
    # 调用人脸属性识别接口
    if options =="":
        result = client.detect(get_file_content(picName))
    else:
        result = client.detect(get_file_content(picName), options)
    print(result)
    end = datetime.datetime.now()
    print("processs is end with process duration time is: "+str(end-begin))

#注册人脸用于识别人的脸
def registerFaceForUser(uid,userInfo,groupId,picPath):
    info= client.addUser(
        uid,
        userInfo,
        groupId,
        get_file_content(picPath)
    )
    print(str(info))


registerFaceForUser("huge","hege is a supper star!","baibu","F:\\python\\huge\\huge-1.jpg")
registerFaceForUser("huge","hege is a supper star!","baibu","F:\\python\\huge\\huge-2.jpg")
registerFaceForUser("huge","hege is a supper star!","baibu","F:\\python\\huge\\huge-3.jpg")
registerFaceForUser("huge","hege is a supper star!","baibu","F:\\python\\huge\\huge-4.jpg")
registerFaceForUser("huge","hege is a supper star!","baibu","F:\\python\\huge\\huge-5.jpg")
registerFaceForUser("huge","hege is a supper star!","baibu","F:\\python\\huge\\huge-6.jpg")
registerFaceForUser("huge","hege is a supper star!","baibu","F:\\python\\huge\\huge-7.jpg")
registerFaceForUser("huge","hege is a supper star!","baibu","F:\\python\\huge\\huge-8.jpg")
registerFaceForUser("huge","hege is a supper star!","baibu","F:\\python\\huge\\huge-9.jpg")
registerFaceForUser("huge","hege is a supper star!","baibu","F:\\python\\huge\\huge-10.jpg")

registerFaceForUser("yangyang","hege is a supper star!","babu","F:\\python\\yangyang\\yang-1.jpg")
registerFaceForUser("yangyang","hege is a supper star!","babu","F:\\python\\yangyang\\yang-2.jpg")
registerFaceForUser("yangyang","hege is a supper star!","babu","F:\\python\\yangyang\\yang-3.jpg")
registerFaceForUser("yangyang","hege is a supper star!","babu","F:\\python\\yangyang\\yang-4.jpg")
registerFaceForUser("yangyang","hege is a supper star!","babu","F:\\python\\yangyang\\yang-5.jpg")
registerFaceForUser("yangyang","hege is a supper star!","babu","F:\\python\\yangyang\\yang-6.jpg")



##检测人脸是否是授权的人物
def recognizeFaceForOne(groupId,picPath):
    options = {
        'user_top_num': 1,
        'face_top_num': 1,
    }
    info = client.identifyUser(
        groupId,
        get_file_content(picPath),
        options
    )
    print(str(info))

print(str(datetime.datetime.now()))
recognizeFaceForOne("babu","F:\\python\\myface\\huge01.jpg")
print(str(datetime.datetime.now()))
recognizeFaceForOne("babu","F:\\python\\myface\\yangyang03.jpg")
print(str(datetime.datetime.now()))
recognizeFaceForOne("babu","F:\\python\\myface\\yang-21.jpg")
print(str(datetime.datetime.now()))








