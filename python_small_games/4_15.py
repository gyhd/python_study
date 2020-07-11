# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

df = pd.read_excel(r"C:\Users\gyhd\Desktop\i_nuc.xlsx", 'Sheet7')
df['sum'] = df.体育 + df.英语 + df.数分 + df.高代 + df.解几 + df.军训


# 对总分分层
bins = [350, 425, 460]
labels = ['low', 'high']
df['group'] = pd.cut(df['sum'], bins, right=False, labels=labels)
print(df.group.value_counts())
# # 对“总分”，按照“班级”为行，“性别”为列，用求和作为统计函数进行交叉分析
# df_pt = df.pivot_table(values=['sum'], index=['班级'],
#                        columns=['性别'], aggfunc=[np.sum])
# print(df_pt)
# 对“高代”和“数分”进行相关性分析
# df_gs = df['高代'].corr(df['数分'])
# print(df_gs)
# 计算相关系数
a = df.loc[:, ['英语', '体育', '军训', '数分']].corr()
print(a)
# 计算单列与其他列的相关系数
b = df.corr()['数分']
print(b)



a, b = 1, 1

for i in range(3,100):
    a, b = b, a+b
    print(a)
    
    
    
for i in range(200):
    if i%5 == 0:
        print(i)


for i in range(2, 100):
    for j in range(2, i):
        if i % j == 0:
            pass
        print(i)


for i in range(1, 10):
    for j in range(1, i):
        print("%d * %d = %d"%(i, j, i*j), end = '\n')

























