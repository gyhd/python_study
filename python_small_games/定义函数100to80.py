# -*- coding: utf-8 -*-


# 定义函数从c加到b，和为s
def plus_to(b, s, c):
    # 将c赋值给a，s与a逐步相加，c的作用是后面的print
    a = c
    while a >= b:
        s += a
        if a == s:
            pass
        else:
            print("from %d to %d equals:%d" % (c, a, s))
        a -= 1  # 步长为1
    return


plus_to(80, 0, 1000)






