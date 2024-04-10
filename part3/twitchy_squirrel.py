import sys

import numpy as np
import pygame

from config import config
from environment import Suika

rng = np.random.default_rng(1)
screen = pygame.display.set_mode((config.screen.width, config.screen.height))
pygame.display.set_caption("PySuika")
clock = pygame.time.Clock()
env = Suika(tag=1)

while True:
    if pygame.event.peek():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    pi_action = rng.random(4)
    action = np.argmax(pi_action)
    env.step(action)
    env.draw(screen)
    pygame.display.update()
    clock.tick(config.screen.fps)

