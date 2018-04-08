import pygame
import sys
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
        self.radius = self.rect.width / 2

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


def main():
    pygame.init()

    ball_image = "grey_ball.png"
    bg_image = "background.jpg"

    running = True

    # 添加背景音乐
    pygame.mixer.music.load('kiojin.ogg')
    pygame.mixer.music.play()

    # 添加音效
    loser_sound = pygame.mixer.Sound('loser.wav')
    winner_sound = pygame.mixer.Sound('winner.wav')
    laugh_sound = pygame.mixer.Sound('laugh.wav')
    hole_sound = pygame.mixer.Sound('hole.wav')

    # 音乐播放完时，游戏结束
    GAMEOVER = USEREVENT
    pygame.mixer.music.set_endevent(GAMEOVER)

    # 根据背景图片决定窗口尺寸
    bg_size = width, height = 1280, 720
    screen = pygame.display.set_mode(bg_size)
    pygame.display.set_caption('Play the ball')

    background = pygame.image.load(bg_image).convert()

    balls = []  # 存放所有小球的对象
    group = pygame.sprite.Group()   # 创建碰撞组

    # 创建五个小球
    for i in range(5):
        position = randint(0, width-50), randint(0, height-50)     # 球的宽度 50x50
        speed = [randint(-10, 10), randint(-10, 10)]
        ball = Ball(ball_image, position, speed, bg_size)    # 实例化
        while pygame.sprite.spritecollide(ball, group, False, pygame.sprite.collide_circle):
            ball.rect.left, ball.rect.top = randint(0, width-50), randint(0, height-50)
        balls.append(ball)
        group.add(ball)

    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit()
            elif event.type == GAMEOVER:
                loser_sound.play()
                pygame.time.delay(2000)
                laugh_sound.play()
                running = False

        screen.blit(background, (0, 0))

        for each in balls:      # 每次调用小球
            each.move()
            screen.blit(each.image, each.rect)

        for each in group:
            group.remove(each)

            # spritecollide(sprite, group, dokill, collide=None)
            # sprite -> 指定被检测的精灵
            # group -> 指定被碰撞的列表  sprite.Group
            # dokill -> 是否删除组中检测到碰撞的精灵
            # collide -> 特殊定制(None检测rect属性)
            if pygame.sprite.spritecollide(each, group, False, pygame.sprite.collide_circle):
                each.speed[0] = -each.speed[0]
                each.speed[1] = -each.speed[1]

            group.add(each)

        pygame.display.flip()
        clock.tick(30)


if __name__ == '__main__':
    main()
