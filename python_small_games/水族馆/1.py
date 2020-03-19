from sprites import *  # 从精灵模块导入所有命令
import random


screen = Screen()  # 新建屏幕
screen.setup(800, 600)  # 设置屏幕宽高
screen.bgpic('sea2.png')  # 设置背景图
screen.titlebar(False)  # 关闭标题栏
screen.draggable()  # 设置按中键可拖动窗口
screen._root.wm_attributes('-alpha', 0.9)  # 设置窗口透明度为0.9

fish1 = Sprite('fish')
fish = Sprite('dog')  # 新建小鱼角色
fish.rotatemode(1)  # 设置为左右翻转模式0,1,2
fishes = [fish]  # 新建列表

# 生成大小不一的15条小鱼,都添加到列表
for x in range(15):  # 重复15次
    f = fish.clone()  # 克隆一条鱼
    s = max(0.5, random.random())  # 设定一个从0.5到1.0(不含1.0)的数据
    # f.scale(s)  # 缩放小鱼
    f.randomgoto(-300, 300, -200, 200)  # 移到一个矩形范围left,right,bottom,top
    f.randomheading()  # 面向随机的一个方向
    fishes.append(f)  # 添加到fishes列表

fish.play('mp3.wav', loop=True)  # 循环播放声音
while True:  # 重复执行
    for f in fishes:  # 列表中的每条鱼
        f.fd(3)  # 前进3个单位
        if random.randint(1, 40) == 1: f.right(180)
        f.nextcostume()  # 下一个造型
        f.bounce_on_edge()  # 碰到边缘就反弹
    time.sleep(0.001)  # 等待0.001秒
    screen.update()  # 屏幕更新显示

