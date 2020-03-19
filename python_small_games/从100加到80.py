c = 100
a = c
s = 0
while a > 79:
    s += a
    if a == s:
        pass
    else:
        print("from %d to %d equals:%d" % (c, a, s))
    a -= 1

