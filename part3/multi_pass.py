import sys

import numpy as np
import pygame

from config import config
from environment import Suika

sw, sh = config.screen.width, config.screen.height
sw2, sh2 = sw // 2, sh // 2

rng = np.random.default_rng(1)
screen = pygame.display.set_mode((sw, sh))
pygame.display.set_caption("PySuika")
clock = pygame.time.Clock()

env1 = Suika(tag=1)
env2 = Suika(tag=2)
env3 = Suika(tag=3)
env4 = Suika(tag=4)
sur1 = pygame.Surface((sw, sh))
sur2 = pygame.Surface((sw, sh))
sur3 = pygame.Surface((sw, sh))
sur4 = pygame.Surface((sw, sh))

while True:
    if pygame.event.peek():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    action = rng.random(4)

    env1.step(action)
    env2.step(action)
    env3.step(action)
    env4.step(action)

    env1.draw(sur1)
    env2.draw(sur2)
    env3.draw(sur3)
    env4.draw(sur4)

    screen.blit(pygame.transform.scale(sur1, size=(sw2, sh2)), dest=(0, 0))
    screen.blit(pygame.transform.scale(sur2, size=(sw2, sh2)), dest=(sw2, 0))
    screen.blit(pygame.transform.scale(sur3, size=(sw2, sh2)), dest=(0, sh2))
    screen.blit(pygame.transform.scale(sur4, size=(sw2, sh2)), dest=(sw2, sh2))

    pygame.display.update()
    clock.tick(config.screen.fps)
