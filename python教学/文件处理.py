# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 15:26:45 2019

@author: Maibenben
"""
"""
import pickle#同样作用的还有marshal，用法也一样

d = dict(name = 'python',age = 20,sex = 'male')

print (pickle.dumps(d))#对文件进行序列化
print(d)

with open('a.txt','wb') as f:
    pickle.dump(d,f)#对文件进行序列化
print (f)


a = b'\x80\x03}q\x00(X\x04\x00\x00\x00nameq\x01X\x06\x00\x00\x00pythonq\x02X\x03\x00\x00\x00ageq\x03K\x14X\x03\x00\x00\x00sexq\x04X\x04\x00\x00\x00maleq\x05u.'

print (pickle.loads(a))#对文件进行反序列化

f = open('a.txt','rb')
d = pickle.load(f)#对文件进行反序列化
f.close()
print(d)


import json

class Person(object):
    def __init__(self,name,age):
        self.name = name
        self.age = age
        
def personDict(std):
    return {
            'name':std.name,
            'age':std.age
            }

s = Person("python","20")
print (s)
print (json.dumps(s,default = personDict))
#使用lambda函数json化
print (json.dumps(s,default = lambda obj:obj.__dict__))

"""




import sys
import json

def get_info(con_file):
    try:
        with open(con_file) as f:
            conf = json.load(f)
        #输入想要查找的参数
        if 'hostname' in conf:
            name = conf['hostname']
            print('got the hostname is %s'%name)
    except:
        print('program is error')

def main(argv):
    if len(argv) < 2:
        sys.stderr.write("Usage: %s <config_file> \n",(argv[0]))
        return -1
    
    config_file = argv[1]
    get_info(config_file)
    
if __name__ == "__main__":
    sys.exit(main(sys.argv))








