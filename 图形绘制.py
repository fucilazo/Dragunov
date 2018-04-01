import pygame
import sys
import math
from pygame.locals import *

pygame.init()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

size = width, height = 640, 480
screen = pygame.display.set_mode(size)
pygame.display.set_caption('drawing')

points = [(200, 75), (300, 25), (400, 75), (450, 25), (450, 125), (400, 75), (300, 125)]    # 多边形点
position = size[0]//2, size[1]//2   # 圆心
moving = False     # 鼠标拖拽移动状态

font = pygame.font.Font(None, 20)
line_height = font.get_linesize()

clock = pygame.time.Clock()


# 显示鼠标坐标
def mouseCoordinate():
    screen.blit(font.render(str(pygame.mouse.get_pos()), True, (0, 0, 0)), (0, 0))  # 内容，是否开启抗锯齿，颜色，（背景色），位置


# 绘制矩形
def drawRect():
    pygame.draw.rect(screen, BLACK, (50, 60, 150, 50), 0)
    pygame.draw.rect(screen, BLACK, (250, 60, 150, 50), 1)
    pygame.draw.rect(screen, BLACK, (450, 60, 150, 50), 10)


# 绘制多边形
def drawPolygon():
    pygame.draw.polygon(screen, GREEN, points, 0)


# 绘制圆形（circle(Surface, Color, pos, radius, width)）
def drawCircle():
    pygame.draw.circle(screen, RED, position, 25, 1)
    pygame.draw.circle(screen, GREEN, position, 75, 1)
    pygame.draw.circle(screen, BLUE, position, 125, 1)


# 绘制椭圆（ellipse(Surface, color, Rect, width)）
def drawEllipse():
    pygame.draw.ellipse(screen, BLACK, (100, 100, 400, 100), 1)
    pygame.draw.ellipse(screen, BLACK, (220, 50, 200, 200), 1)


# 绘制弧线（arc(Surface, color, Rect, start_angle,stop_angle, width = 1)）
def drawArc():
    pygame.draw.arc(screen, BLACK, (100, 100, 440, 100), 0, math.pi, 1)
    pygame.draw.arc(screen, BLACK, (220, 50, 200, 200), math.pi, math.pi * 2, 1)


# 绘制线段（line(Surface, color, start_pos, end_pos, width=1)
#         lines(Surface, color, closed, pointlist, width=1)）        closed->是否首尾相连
def drawLines():
    pygame.draw.lines(screen, GREEN, 1, points, 1)
    pygame.draw.line(screen, BLACK, (100, 200), (540, 250), 1)


# 绘制抗锯齿线段（aaline(Surface, color,startpos, endpos, blend=1)      blend->是否通过混合背景的阴影实现抗锯齿
#              aalines(Surface, color, closed, pointlist, blend=1)）
    pygame.draw.aaline(screen, BLACK, (100, 250), (540, 300), 1)
    pygame.draw.aaline(screen, BLACK, (100, 300), (540, 350), 0)


while True:
    screen.fill(WHITE)

    mouseCoordinate()
    # drawRect()
    # drawPolygon()
    # drawCircle()
    # drawEllipse()
    # drawArc()
    # drawLines()

    for event in pygame.event.get():
        if event.type == QUIT:
            sys.exit()

        if event.type == MOUSEBUTTONDOWN:
            if event.button == 1:   # 1:左键 2:中键 3:右键 4:滚轮向上 5:滚轮向下
                moving = True

        if event.type == MOUSEBUTTONUP:
            if event.button == 1:
                moving = False

    if moving:
        position = pygame.mouse.get_pos()

    pygame.display.flip()   # 显示
    clock.tick(30)  # 30帧
