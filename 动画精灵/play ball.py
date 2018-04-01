import pygame
import sys
from pygame.locals import *
from random import *


class Ball(pygame.sprite.Sprite):
    def __init__(self, image, position, speed):     # 初始化
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.image.load(image).convert()
        self.rect = self.image.get_rect()           # 获取尺寸
        self.rect.left, self.rect.top = position    # 获取位置
        self.speed = speed                          # 获取速度


def main():
    pygame.init()

    ball_image = "grey_ball.png"
    bg_image = "background.jpg"

    running = True

    bg_size = width, height = 1024, 681
    screen = pygame.display.set_mode(bg_size)
    pygame.display.set_caption('Play the ball')

    background = pygame.image.load(bg_image).convert()

    balls = []

    for i in range(5):
        position = randint(0, width-50), randint(0, height-50)     # 球的宽度 50x50
        speed = [randint(-10, 10), randint(-10, 10)]
        ball = Ball(ball_image, position, speed)    # 实例化
        balls.append(ball)

    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit()

        screen.blit(background, (0, 0))

        for each in balls:
            screen.blit(each.image, each.rect)

        pygame.display.flip()
        clock.tick(30)


if __name__ == '__main__':
    main()
