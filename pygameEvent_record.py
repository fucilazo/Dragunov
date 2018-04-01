import pygame
import sys

pygame.init()

size = width, height = 600, 400
screen = pygame.display.set_mode(size)
pygame.display.set_caption('RecorD')
bg = (0, 0, 0)

font = pygame.font.Font(None, 20)
line_height = font.get_linesize()
position = 0

screen.fill(bg)

# f = open('record.txt', 'w')

while True:
    for event in pygame.event.get():
        # f.write(str(event) + '\n')

        if event.type == pygame.QUIT:
            # f.close()
            sys.exit()

        screen.blit(font.render(str(event), True, (0, 255, 0)), (0, position))
        position += line_height

        if position > height:
            position = 0
            screen.fill(bg)

    pygame.display.flip()
