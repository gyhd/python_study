# -*- coding: utf-8 -*-


import math

N, L = map(int, input().split())
i = L
while i <= 100:
    a = (2 * N - i*(i - 1)) / (2 * i)
    if math.ceil(a) == a:
        for j in range(i- 1):
            print(int(a), end=" ")
            a += 1
        print(int(a))
        break
    i += 1
if i == 101:
    print("No")












