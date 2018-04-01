import pygame
import sys
from pygame.locals import *


# 初始化pygame
pygame.init()

size = width, height = 600, 400
speed = [-2, 1]
bg = (255, 255, 255)    # RGB

fullscreen = False

# 创建指定大小的窗口
screen = pygame.display.set_mode(size, RESIZABLE)   # RESIZABLE-->窗口可修改
# 设置窗口标题
pygame.display.set_caption('TITLE')


# 加载图片
opic = pygame.image.load('aaa.jpg')
pic = opic
# 获得图像的位置矩形（矩形对象）
opic_rect = opic.get_rect()
position = pic_rect = opic_rect

# 设置放大缩小的比率
ratio = 1.0

# 图片调转
l_head = pic
r_head = pygame.transform.flip(pic, True, False)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:   # 点击X按钮
            sys.exit()

        # 重新控制移动方向
        if event.type == KEYDOWN:       # 按键监听
            if event.key == K_LEFT:
                pic = l_head
                speed = [-3, 0]
            if event.key == K_RIGHT:
                pic = r_head
                speed = [3, 0]
            if event.key == K_UP:
                speed = [0, -3]
            if event.key == K_DOWN:
                speed = [0, 3]

            # 全屏
            if event.key == K_F11:
                fullscreen = not fullscreen
                if fullscreen:
                    size = width, height = 1366, 768
                    screen = pygame.display.set_mode(size, FULLSCREEN | HWSURFACE)  # HWSURFACE-->硬件加速
                else:
                    size = width, height = 600, 400
                    screen = pygame.display.set_mode(size)

            # 放大缩小图片（=、-），空格恢复原始尺寸
            if event.key == K_EQUALS or event.key == K_MINUS or event.key == K_SPACE:
                # 最大只能放大一倍，缩小50%
                if event.key == K_EQUALS and ratio < 2:
                    ratio += 0.1
                if event.key == K_MINUS and ratio > 0.5:
                    ratio -= 0.1
                if event.key == K_SPACE:
                    ratio = 1.0
                pic = pygame.transform.smoothscale(opic, (int(opic_rect.width * ratio), int(opic_rect.height * ratio)))

                l_head = pic
                r_head = pygame.transform.flip(pic, True, False)
        # 用户调整窗口尺寸
        if event.type == VIDEORESIZE:
            size = event.size
            width, height = size
            print(size)
            screen = pygame.display.set_mode(size, RESIZABLE)   # 重新打印屏幕

    # 移动图像
    position = position.move(speed)

    if position.left < 0 or position.right > width:
        # 翻转图像
        pic = pygame.transform.flip(pic, True, False)
        # 反向移动
        speed[0] = -speed[0]

    if position.top < 0 or position.bottom > height:
        speed[1] = -speed[1]

    # 填充背景
    screen.fill(bg)
    # 更新图像
    screen.blit(pic, position)
    # 更新界面
    pygame.display.flip()   # 双缓冲（避免闪烁）
    # 延迟10毫秒
    pygame.time.delay(10)
