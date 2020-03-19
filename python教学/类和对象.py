# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 19:22:47 2019

@author: Maibenben
"""
"""
class People:
    def __init__(self,n,a):
        self.__name=n
        self.age=a
        
    def get_name(self):
        return self.__name
    
    def set_age(self,name):
        self.__name=name
        
    def speak(self):
        print("%s,I am %d years old"%(self.__name,self.age))

p=People('fiona',20)
p.speak()
        
p.set_age('anna')
p.speak()



class Animal:
    
    def eat(self):
        print("%s,eat"%self.name)
    def drink(self):
        print("%s,drink"%self.name)

class cat(Animal):
    
    def __init__(self,name):
        self.name=name
    def speak(self):
        print("%s,wangwangwang"%self.name)

a1=cat('fiona')
a1.eat()
a1.speak()




class Animal(object):
    def run(self):
        print('run')
        
class Cat(Animal):
    def run(self):
        print('running')
        
class Dog():
    def run(self):
        print('running...')

def run_twice(Animal):
    Animal.run()
    Animal.run()

run_twice(Dog())
run_twice(Cat())



a = input('请输入一个字符:')
b = int(input('请输入一个ASCII码:'))

print(a + '对应的字符是',ord(a))
print(b, '的ASCII码是',chr(b))

"""

a = open('c.txt','w')
b = a.write('gyhd')
print(b)

