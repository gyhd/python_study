
def move(man, sep):
    """
    将man列表向左移动sep单位，最左边的元素向列表后面添加，
    相当于队列顺时针移动
    """
    for i in range(sep):
        # 将下一轮的前两个数字顺时针到man的后面
        item = man.pop(0)
        print('item: ', item)
        man.append(item)


def play(man, sep, rest):
    """
    man：玩家个数
    sep：杀死数到的第几个人
    rest：剩余的数量
    """
    print('')
    man = [i for i in range(1, man + 1)]  # 初始化玩家队列
    print('player groups:', man)
    sep -= 1  # 数两个数，到第三个人就自杀
    s = 0

    while len(man) > rest:
        print('这是第%d轮：' %s)
        s += 1
        move(man, sep)
        print('kill：', man.pop(0))

    return man


servive = play(41, 3, 2)
print('', servive)


# play(41, 3, 2)


