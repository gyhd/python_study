# -*- coding: utf-8 -*-
a = []
b = []
for i in range(1, 101):
    b.append(i)
    for j in range(2, i):      
        if i % j == 0:
            break
    else:
        a.append(i)


print(a)
print(b)

c = []
for k in b:
    for l in a:
        if k == l:
            break
    else:
        c.append(k)

            
print(c)


L=[]
for x in range(100):
    if x<2:
        continue
    for i in range(2,x):
        if x%i==0:
            break    
    else:   #走到此处，x一定是素数
        L.append(x)
print("100以内的全部素数有：",L)


import math

a = math.ceil(4.12)  # 取整数，整数部分+1
print('a=',a)

b = math.copysign(-2, 3)  # 将后者的符号给前者
print('b=',b)

c = math.cos(math.pi/3)  # 求cos（x），sin（x），tan（x）一样
print('c=',c)

d = math.degrees(math.pi/3)  # 将数转化成弧度数
print('d=',d)

e = math.pi  # pi
print('e=',e)

f = math.e  # e
print('f=',f)

g = math.exp(2)  # e的x次方
print('g=',g)

h = math.expm1(2)  #e的x次方-1
print('h=',h)

i = math.fabs(-0.3)  # 取绝对值
print('i=',i)

j = math.factorial(4)  # 返回x的阶乘
print('j=',j)

k = math.floor(4.9)  # 取整数，舍去小数点后的
print('k=',k)

l = math.fmod(5, 3)  # 取余数，结果为浮点值
print('l=',l)

m = math.fsum((1, 2, 3, 4))  # 对里面内容求和
print('m=',m)

n = math.gcd(12, 16)  # 求最大公约数
print('n=',n)

a1 = math.hypot(3, 4)  # 两者的平方和
print('a1=',a1)

a2 = math.log(math.e)  # 取对数，底数为e
print('a2=',a2)

a3 = math.log(32, 2)  # 取对数，底数为2
print('a3=',a3)

a4 = math.log10(100)  # 取对数，底数为10
print('a4=',a4)

a5 = math.log2(32)  # 取对数，底数为2
print('a5=',a5)

a6 = math.modf(math.e)  # 将小数，整数部分，组合成一个元组
print('a6=',a6)

a7 = math.pow(2, 3)  # x的y次方
print('a7=',a7)

a8 = math.radians(180)  # 将弧度数转化成数值
print('a8=',a8)

a9 = math.sqrt(16)  # 取算术平方根
print('a9=',a9)










