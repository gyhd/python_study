import pandas as pd
import numpy as np

df = pd.read_excel(r"C:\Users\gyhd\Desktop\i_nuc.xlsx", sheet_name='Sheet7')

# 查看前十个数据
# print(df.head(10))
# print(df.高代.describe())
# print("平均值为：", df.高代.mean())
# print("最大值为：", df.高代.max())
# print("规则为：", df.高代.size)
# print("最小值为：", df.高代.min())
# print("方差为：", df.高代.std())
# print(df.groupby('性别')['体育', '军训', '英语'].mean())
# print(df.groupby('性别')['体育', '军训', '英语'].agg({np.sum,np.mean}))
# df.zongfen = df.体育 + df.英语 + df.数分 + df.高代 + df.解几 + df.军训

print(df.军训.describe())
# print(df.军训)

print(df.groupby('班级')['数分', '高代', '英语'].mean())

