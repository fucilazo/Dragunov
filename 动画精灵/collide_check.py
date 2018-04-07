import pygame
import sys
import math
from pygame.locals import *
from random import *


class Ball(pygame.sprite.Sprite):
    def __init__(self, image, position, speed, bg_size):     # 初始化
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.image.load(image).convert_alpha()
        self.rect = self.image.get_rect()           # 获取尺寸
        self.rect.left, self.rect.top = position    # 获取位置
        self.speed = speed                          # 获取速度
        self.width, self.height = bg_size[0], bg_size[1]

    def move(self):
        self.rect = self.rect.move(self.speed)
        if self.rect.right < 0:             # 小球右边界 碰到 屏幕左边界
            self.rect.left = self.width     # 小球从右侧出来
        elif self.rect.left > self.width:   # 小球左边界 碰到 屏幕右边界
            self.rect.right = 0             # 小球从左侧出来
        elif self.rect.bottom < 0:          # 小球下边界 碰到 屏幕上边界
            self.rect.top = self.height     # 小球从下侧出来
        elif self.rect.top > self.height:   # 小球上边界 碰到 屏幕下边界
            self.rect.bottom = 0            # 小球从上侧出来


# 碰撞检测原理
def collide_check(item, target):    # item、target -> 球1、球2
    col_balls = []  # 存入和'item'发生碰撞的球
    for each in target:
        distance = math.sqrt(
            math.pow((item.rect.center[0] - each.rect.center[0]), 2) +
            math.pow((item.rect.center[1] - each.rect.center[1]), 2))
        if distance <= (item.rect.width + each.rect.width) / 2:
            col_balls.append(each)

    return col_balls


def main():
    pygame.init()

    ball_image = "grey_ball.png"
    bg_image = "background.jpg"

    running = True

    bg_size = width, height = 1024, 681
    screen = pygame.display.set_mode(bg_size)
    pygame.display.set_caption('Play the ball')

    background = pygame.image.load(bg_image).convert()

    balls = []  # 存放所有小球的对象

    BALL_NUM = 5    # 小球数量

    for i in range(BALL_NUM):   # 创建小球
        position = randint(0, width-50), randint(0, height-50)     # 球的宽度 50x50
        speed = [randint(-10, 10), randint(-10, 10)]
        ball = Ball(ball_image, position, speed, bg_size)    # 实例化
        while collide_check(ball, balls):   # 如果两个小球出生的时候重叠了
            ball.rect.left, ball.rect.top = randint(0, width-50), randint(0, height-50)
        balls.append(ball)

    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit()

        screen.blit(background, (0, 0))

        for each in balls:      # 每次调用小球
            each.move()
            screen.blit(each.image, each.rect)

        for i in range(BALL_NUM):
            item = balls.pop(i)     # 被检测碰撞的小球
            if collide_check(item, balls):  # 碰撞后速度取反
                item.speed[0] = -item.speed[0]
                item.speed[1] = -item.speed[1]
            balls.insert(i, item)   # 将小球放回

        pygame.display.flip()
        clock.tick(30)


if __name__ == '__main__':
    main()
