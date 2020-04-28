
import numpy as np
import pandas as pd

# np.random.randn代表生成四行五列随机数据
# index代表每一行的属性
# columns代表每一列的属性
df18 = pd.DataFrame(np.random.randn(4, 5), index=[
        'apple', 'banana', 'orange', 'pear'], columns=[
                'name', 'value1', 'value2', 'value3', 'value4'])

print(df18)

# 每一列的属性
col_name = df18.columns.values
print("列属性:", col_name)

# 每一行的属性
row_name = df18.index.values
print("行属性:", row_name)

# 获取第一列前三行信息
df1_3 = df18['name'][0:3]
print(df1_3)

# 获取前三列的前三行内容
df3_3 = df18[df18.columns[0:3]][0:3]
print(df3_3)

# 添加新的一列, 5代表位置
df_grade = df18.insert(5, 'grade', [1,2,3,4])
print(df_grade)

# 添加新的一行
df18.loc['grape'] = [10, 20, 5, 4, 6, 8]
print(df18)

