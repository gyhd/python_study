#
# import pandas as pd
# import numpy as np
#
# df = pd.read_csv(r"C:\Users\gyhd\Desktop\F_test.csv")
#
# print(df.passengers.describe())
# # print(df)
#
# df1 = df[df['passengers'] > 400]
# df1.to_csv('O_test.txt', index=False, sep=' ', encoding='utf_8_sig')
#
# # print(df1)
#
# # # 对“总分”，按照“班级”为行，“性别”为列，用求和作为统计函数进行交叉分析
# df_pt = df.pivot_table(values=['passengers'], index=['year'],
#                        columns=['month'], aggfunc=[np.sum])
# print(df_pt)
#
#
# df2 = df['year'].corr(df['passengers'])
# print(df2)
#


import seaborn as sns

titanic = pd.read_csv(r"C:\Users\gyhd\Desktop\titanic.csv")  # 读取数据集, 返回 DataFrame
#
# tit_age = tit['age'].dropna()
# tit_age.to_csv('tit_age.csv', index=False, sep=' ', encoding='utf_8_sig')
# tit_age = pd.read_csv("tit_age.csv")
# sns.set_style('dark')  # 该图使用黑色为背景色
# sns.distplot(tit_age['age'], kde=True)
# # sns.distplot(tit1['age'], kde=True)  # 不显示密度曲线
# sns.axlabel('Birth number', 'Frequency') # 设置X轴和Y轴的坐标含义
# sns.plt.show()
















