#---第1步---导入模块---
from turtle import *
from random import random,randint

#---第2步---初始化定义---
#---定义屏幕，窗口大小，标题，背景颜色
screen = Screen()
#---大一点效果好一点---
width ,height = 800, 600
screen.setup(width,height)
screen.title('浪漫星空')
screen.bgcolor("black")
#设置或返回以毫秒为单位的绘制延迟，延迟越大，绘图越慢
screen.delay(0)


#---第3步---定义3种不同颜色的星球，大小、速度、位置、形状不同---
#shape():设置乌龟的图形形状，取值:“arrow”, “turtle”, “circle”, “square”, “triangle”, “classic”
#---星球---白色星星---
t = Turtle(visible = False,shape='circle')
t.pencolor("white")
#海龟的颜色，也就是飞动的星球的颜色
t.fillcolor("blue")
t.penup()
#旋转角度
t.setheading(-10)
#坐标是随机的
t.goto(width/2,randint(-height/2,height/2))

#---星球2---绿色远处小星星---
t2 = Turtle(visible = False,shape='turtle')
#海龟的颜色，也就是飞动的星球的颜色
t2.fillcolor("green")
t2.penup()
t2.setheading(-50)
#坐标是随机的
t2.goto(width,randint(-height,height))

#---星球3---近的红色恒星---
t3 = Turtle(visible = False,shape='circle')
#海龟的颜色，也就是飞动的星球的颜色
t3.fillcolor("red")
t3.penup()
t3.setheading(-90)
#坐标是随机的
t3.goto(width*2,randint(-height*2,height*2))

#---第4步---定义星球列表---用于存放---
stars = []
stars2 = []
stars3 = []

#---第5步---定义3种星球的大小、速度、位置并存放各自列表中---
#---注意200为画200个各自星球就退出，注意太多了要卡死的---
for i in range(200):
    star = t.clone()
    #决定星球的大小
    s= random()/3
    star.shapesize(s,s)
    star.speed(int(s*10))
    #随机产生坐标
    star.setx(width/2 + randint(1,width))
    star.sety(randint(-height/2,height/2))
    star.showturtle()
    stars.append(star)

for i in range(200):
    star2 = t2.clone()
    #决定星球的大小
    s2= random()/2
    star2.shapesize(s2,s2)
    star2.speed(int(s*10))
    star2.setx(width/2 + randint(1,width))
    star2.sety(randint(-height/2,height/2))
    star2.showturtle()
    stars2.append(star2)

for i in range(200):
    star3 = t3.clone()
    #决定星球的大小
    s3= random()*5
    star3.shapesize(10*s3,10*s3)
    star3.speed(int(s3*10))
    star3.setx(width*2 + randint(1,width))
    star3.sety(randint(-height*2,height*2))
    star3.showturtle()
    stars3.append(star3)

#---第6步---游戏循环---各自星球的启动
while True:
    for star in stars:
        star.setx(star.xcor() - 3 * star.speed())
        if star.xcor()<-width/2:
            star.hideturtle()
            star.setx(width/2 + randint(1,width))
            star.sety( randint(-height/2,height/2))
            star.showturtle()

    for star2 in stars2:
        star2.setx(star2.xcor() - 3 * star2.speed())
        if star2.xcor()<-width/2:
            star2.hideturtle()
            star2.setx(width/2 + randint(1,width))
            star2.sety( randint(-height/2,height*2))
            star2.showturtle()

    for star3 in stars3:
        star3.setx(star3.xcor() - 3 * star3.speed())
        if star3.xcor()<-width*2:
            star3.hideturtle()
            star3.setx(width*2 + randint(1,width))
            star3.sety( randint(-height*2,height*2))
            star3.showturtle()