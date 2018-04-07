import pygame, sys
from math import *
pygame.init()
screen = pygame.display.set_mode((800,700),0,32)
missile = pygame.image.load('red_pointer.png').convert_alpha()
x1, y1 = 100, 600   # 导弹的初始发射位置
velocity = 800      # 导弹速度
time = 1/1000       # 每个时间片的长度
clock = pygame.time.Clock()
old_angle = 0
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
    clock.tick(300)
    x, y = pygame.mouse.get_pos()               # 获取鼠标位置，鼠标就是需要打击的目标
    distance = sqrt(pow(x1-x, 2)+pow(y1-y, 2))  # 两点距离公式
    section = velocity*time                     # 每个时间片需要移动的距离
    sina = (y1-y)/distance
    cosa = (x-x1)/distance
    angle = atan2(y-y1, x-x1)                   # 两点线段的弧度值
    x1, y1 = (x1+section*cosa, y1-section*sina)
    d_angle = degrees(angle)                    # 弧度转角度
    screen.blit(missile, (x1-missile.get_width(), y1-missile.get_height()/2))
    dis_angle = d_angle-old_angle               # dis_angle就是到下一个位置需要改变的角度
    old_angle = d_angle                         # 更新初始角度
    pygame.display.update()