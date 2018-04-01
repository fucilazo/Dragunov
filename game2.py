import pygame
import sys
from pygame.locals import *

pygame.init()

size = width, height = 800, 600
bg = (255, 255, 255)

clock = pygame.time.Clock()
screen = pygame.display.set_mode(size)
pygame.display.set_caption('chop')

pic = pygame.image.load('test.jpg').convert()     # 转化为像素格式（png使用convert_alpha()）

# 0 -> 未选择，1 -> 选择中，2 -> 完成选择
select = 0
select_rect = pygame.Rect(0, 0, 0, 0)

# 0 -> 未拖拽，1 -> 拖拽中，2 -> 完成拖拽
drag = 0

position = pic.get_rect()
position.center = width // 2, height // 2

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            sys.exit()

        elif event.type == MOUSEBUTTONDOWN:
            if event.button == 1:
                # 第一次点击，选择范围
                if select == 0 and drag == 0:
                    pos_start = event.pos
                    select = 1      # ---选择中
                # 第二次点击，拖拽图像
                elif select == 2 and drag == 0:
                    capture = screen.subsurface(select_rect).copy()
                    cap_rect = capture.get_rect()
                    drag = 1        # ---拖拽中
                # 第三次点击，初始化
                elif select == 2 and drag == 2:
                    select = 0
                    drag = 0
        elif event.type == MOUSEBUTTONUP:
            if event.button == 1:
                # 第一次释放，结束选择
                if select == 1 and drag == 0:
                    pos_stop = event.pos
                    select = 2      # ---结束选择
                # 第二次释放，结束拖拽
                if select == 2 and drag == 1:
                    drag = 2       # ---结束拖拽
    screen.fill(bg)
    screen.blit(pic, position)

    # 实时绘制选择框
    if select:
        mouse_pos = pygame.mouse.get_pos()
        if select == 1:
            pos_stop = mouse_pos
        select_rect.left, select_rect.top = pos_start
        select_rect.width, select_rect.height = pos_stop[0] - pos_start[0], pos_stop[1] - pos_start[1]
        pygame.draw.rect(screen, (0, 0, 0), select_rect, 1)  # draw方法画rect：(位置，颜色，位置，宽度)

    # 拖拽裁剪的图像
    if drag:
        if drag == 1:
            cap_rect.center = mouse_pos     # 实时位置
        screen.blit(capture, cap_rect)      # 绘制拖拽中图像

    pygame.display.flip()

    clock.tick(30)

