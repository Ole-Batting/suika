import pygame

from config import config

pygame.font.init()
score_font = pygame.font.SysFont("noteworthy", 36, True)
over_font = pygame.font.SysFont("noteworthy", 72, True)


def center(label, screen, loc):
    x, y = loc
    half_width = label.get_width() / 2
    half_height = label.get_height() / 2
    loc = (x - half_width, y - half_height)

    screen.blit(label, loc)


def score(val, screen: pygame.Surface):
    label = score_font.render(str(val), True, (255, 230, 128))
    center(label, screen, config.screen.score)


def gameover(screen):
    label = over_font.render("Game Over!", True, (0, 0, 0))
    center(label, screen, config.screen_center)
