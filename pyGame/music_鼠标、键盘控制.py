import pygame
import sys
from pygame.locals import *

pygame.init()
pygame.mixer.init()  # 初始化混音器模块(一般pygame会初始化此模块)

pygame.mixer.music.load('Lilium.ogg')
pygame.mixer.music.set_volume(0.2)
pygame.mixer.music.play()

cat_sound = pygame.mixer.Sound('cat.wav')
cat_sound.set_volume(0.2)
dog_sound = pygame.mixer.Sound('dog.wav')
dog_sound.set_volume(0.2)

bg_size = width, height = 300, 200
screen = pygame.display.set_mode(bg_size)
pygame.display.set_caption('Music Demo')

pause = False
pause_image = pygame.image.load('pause.png').convert_alpha()
play_image = pygame.image.load('play.png').convert_alpha()
pause_rect = pause_image.get_rect()
pause_rect.left, pause_rect.top = (width - pause_rect.width) // 2, (height - pause_rect.height) // 2

clock = pygame.time.Clock()

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            sys.exit()

        if event.type == MOUSEBUTTONDOWN:
            if event.button == 1:   # 左键
                cat_sound.play()
            if event.button == 3:   # 右键
                dog_sound.play()

        if event.type == KEYDOWN:
            if event.key == K_SPACE:
                pause = not pause

    screen.fill((255, 255, 255))

    if pause:
        screen.blit(play_image, pause_rect)
        pygame.mixer.music.pause()
    else:
        screen.blit(pause_image, pause_rect)
        pygame.mixer.music.unpause()

    pygame.display.flip()

    clock.tick(30)
