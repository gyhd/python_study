# -*- coding: utf-8 -*-


import re
import pandas as pd

ip = pd.read_excel("ip.xlsx")

ip['IP'].str.strip()
new_ip = ip['IP'].str.split('.', 1, True)  # 按照'.'进行划分，1代表只分一列出来

print(new_ip)









