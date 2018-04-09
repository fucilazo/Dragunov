import pygame
import sys
import traceback
from pygame.locals import *
from random import *


class Ball(pygame.sprite.Sprite):
    def __init__(self, greyball_image, greenball_image, position, speed, bg_size, target):     # 初始化
        pygame.sprite.Sprite.__init__(self)

        self.greyball_image = pygame.image.load(greyball_image).convert_alpha()
        self.greenball_image = pygame.image.load(greenball_image).convert_alpha()
        self.rect = self.greyball_image.get_rect()      # 获取尺寸
        self.rect.left, self.rect.top = position        # 获取位置
        self.side = [choice([-1, 1]), choice([-1, 1])]  # 方向
        self.speed = speed                              # 获取速度
        self.collide = False                            # 是否碰撞
        self.target = target                            # 使小球停止的target
        self.control = False                            # 是否可以控制
        self.width, self.height = bg_size[0], bg_size[1]
        self.radius = self.rect.width / 2

    def move(self):
        if self.control:
            self.rect = self.rect.move(self.speed)
        else:
            self.rect = self.rect.move((self.side[0] * self.speed[0], \
                                        self.side[1] * self.speed[1]))
        if self.rect.right <= 0:             # 小球右边界 碰到 屏幕左边界
            self.rect.left = self.width     # 小球从右侧出来
        elif self.rect.left >= self.width:   # 小球左边界 碰到 屏幕右边界
            self.rect.right = 0             # 小球从左侧出来
        elif self.rect.bottom <= 0:          # 小球下边界 碰到 屏幕上边界
            self.rect.top = self.height     # 小球从下侧出来
        elif self.rect.top >= self.height:   # 小球上边界 碰到 屏幕下边界
            self.rect.bottom = 0            # 小球从上侧出来

    def check(self, motion):
        if self.target < motion < self.target + 5:
            return True
        else:
            return False


class Glass(pygame.sprite.Sprite):
    def __init__(self, glass_image, mouse_image, bg_size):     # 初始化
        pygame.sprite.Sprite.__init__(self)

        self.glass_image = pygame.image.load(glass_image).convert_alpha()
        self.glass_rect = self.glass_image.get_rect()
        self.glass_rect.left, self.glass_rect.top = \
            (bg_size[0] - self.glass_rect.width) // 2, bg_size[1] - self.glass_rect.height
        self.mouse_image = pygame.image.load(mouse_image).convert_alpha()
        self.mouse_rect = self.mouse_image.get_rect()
        self.mouse_rect.left, self.mouse_rect.top = \
            self.glass_rect.left, self.glass_rect.top
        pygame.mouse.set_visible(False)     # 鼠标不可见


def main():
    pygame.init()

    greyball_image = "grey_ball.png"
    greenball_image = "green_ball.png"
    glass_image = "glass.png"
    mouse_image = "hand.png"
    bg_image = "background.jpg"

    running = True

    # 添加背景音乐
    pygame.mixer.music.load('上海红茶馆.ogg')
    pygame.mixer.music.play()

    # 添加音效
    loser_sound = pygame.mixer.Sound('loser.wav')
    winner_sound = pygame.mixer.Sound('winner.wav')
    laugh_sound = pygame.mixer.Sound('laugh.wav')
    hole_sound = pygame.mixer.Sound('hole.wav')

    # 音乐播放完时，游戏结束
    GAMEOVER = USEREVENT    # 自定义事件
    pygame.mixer.music.set_endevent(GAMEOVER)

    # 根据背景图片决定窗口尺寸
    bg_size = width, height = 1133, 665
    screen = pygame.display.set_mode(bg_size)
    pygame.display.set_caption('Play the ball')

    background = pygame.image.load(bg_image).convert()

    # 黑洞位置(x1-x2,y1-y2)(活动范围)
    hole = [(97, 244, 48, 117), (183, 257, 360, 429), (422, 523, 192, 261), (946, 1015, 131, 230), (752, 828, 353, 422)]

    msgs = []

    balls = []  # 存放所有小球的对象
    group = pygame.sprite.Group()   # 创建碰撞组

    # 创建五个小球
    for i in range(5):
        position = randint(0, width-50), randint(0, height-50)     # 球的宽度 50x50
        speed = [randint(1, 10), randint(1, 10)]    # self.side 为方向
        ball = Ball(greyball_image, greenball_image, position, speed, bg_size, 5*(i+1))    # 实例化
        while pygame.sprite.spritecollide(ball, group, False, pygame.sprite.collide_circle):
            ball.rect.left, ball.rect.top = randint(0, width-50), randint(0, height-50)
        balls.append(ball)
        group.add(ball)

    glass = Glass(glass_image, mouse_image, bg_size)

    motion = 0  # 记录鼠标每秒钟产生的事件数量

    MYTIMER = USEREVENT + 1     # 自定义事件
    pygame.time.set_timer(MYTIMER, 1000)

    # set_repeat(delay, inerval)
    # delay -> 第一次发送事件的延迟时间
    # interva -> 重复发送事件的间隔
    # 如果不带任何参数，表示取消重复发送事件
    pygame.key.set_repeat(100, 100)     # 按键重复响应

    clock = pygame.time.Clock()

    while running:  # 一次循环相当于跑一帧
        for event in pygame.event.get():
            if event.type == QUIT:
                # pygame.quit()   # 解决idle下关闭未响应
                sys.exit()
            elif event.type == GAMEOVER:    # 游戏结束时
                loser_sound.play()
                pygame.time.delay(1000)
                laugh_sound.play()
                running = False
            elif event.type == MYTIMER:     # 改变小球运动的监听
                if motion:
                    for each in group:
                        if each.check(motion):
                            each.speed = [0, 0]
                            each.control = True
                    motion = 0
            elif event.type == MOUSEMOTION:
                motion += 1

            elif event.type == KEYDOWN:
                if event.key == K_w:
                    # 迭代每一个小球
                    for each in group:
                        # 如果可控
                        if each.control:
                            # 加速行走
                            each.speed[1] -= 1  # Y轴向上

                if event.key == K_s:
                    for each in group:
                        if each.control:
                            each.speed[1] += 1  # Y轴向下

                if event.key == K_a:
                    for each in group:
                        if each.control:
                            each.speed[0] -= 1  # X轴向左

                if event.key == K_d:
                    for each in group:
                        if each.control:
                            each.speed[0] += 1  # X轴向右

                if event.key == K_SPACE:
                    for each in group:  # 迭代每个小球
                        if each.control:    # 被控制时
                            for i in hole:
                                if i[0] <= each.rect.left <= i[1] and \
                                   i[2] <= each.rect.top <= i[3]:
                                    hole_sound.play()
                                    each.speed = [0, 0]
                                    group.remove(each)
                                    temp = balls.pop(balls.index(each))
                                    balls.insert(0, temp)
                                    hole.remove(i)
                            if not hole:
                                pygame.mixer.music.stop()
                                winner_sound.play()
                                pygame.time.delay(1000)
                                msg = pygame.image.load('over.jpg')
                                msg_pos = (width - msg.get_width()) // 2, \
                                          (height - msg.get_height()) // 2
                                msgs.append((msg, msg_pos))
                                laugh_sound.play()

        screen.blit(background, (0, 0))
        screen.blit(glass.glass_image, glass.glass_rect)

        glass.mouse_rect.left, glass.mouse_rect.top = pygame.mouse.get_pos()    # 鼠标图片跟随

        # 仅在玻璃面板内
        if glass.mouse_rect.left < glass.glass_rect.left:
            glass.mouse_rect.left = glass.glass_rect.left
        if glass.mouse_rect.right > glass.glass_rect.right:
            glass.mouse_rect.right = glass.glass_rect.right
        if glass.mouse_rect.top < glass.glass_rect.top:
            glass.mouse_rect.top = glass.glass_rect.top
        if glass.mouse_rect.bottom > glass.glass_rect.bottom:
            glass.mouse_rect.bottom = glass.glass_rect.bottom

        screen.blit(glass.mouse_image, glass.mouse_rect)    # 画鼠标

        for each in balls:      # 每次调用小球
            each.move()
            if each.collide:
                each.speed = [randint(1, 10), randint(1, 10)]
                each.collide = False
            if each.control:
                # 画绿色的小球
                screen.blit(each.greenball_image, each.rect)
            else:
                # 画灰色的小球
                screen.blit(each.greyball_image, each.rect)

        # 碰撞检测
        for each in group:
            group.remove(each)
            # spritecollide(sprite, group, dokill, collide=None)
            # sprite -> 指定被检测的精灵
            # group -> 指定被碰撞的列表  sprite.Group
            # dokill -> 是否删除组中检测到碰撞的精灵
            # collide -> 特殊定制(None检测rect属性)
            if pygame.sprite.spritecollide(each, group, False, pygame.sprite.collide_circle):
                # each.speed[0] = -each.speed[0]
                # each.speed[1] = -each.speed[1]
                # each.speed = [randint(1, 10), randint(1, 10)]   # 碰撞后获得随机速度
                each.side[0] = -each.side[0]
                each.side[1] = -each.side[1]
                each.collide = True
                if each.control:
                    each.side[0] = -1
                    each.side[1] = -1
                    each.control = False

            group.add(each)

        for msg in msgs:
            screen.blit(msg[0], msg[1])

        pygame.display.flip()
        clock.tick(30)


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
    except:
        traceback.print_exc()
        pygame.quit()
        input()

