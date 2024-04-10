import numpy as np
import pygame

from config import config
from particle import Particle

rng = np.random.default_rng()


class PreParticle:
    def __init__(self):
        self.x = config.screen.width // 2
        self.n = rng.integers(config.test.low_pp_rng, config.test.high_pp_rng)
        self.radius = config[self.n, "radius"]
        self.sprite = config[self.n, "blit"]
        self.left_lim = config.pad.left + self.radius
        self.right_lim = config.pad.right - self.radius

    def draw(self, screen, wait):
        screen.blit(config.cloud_blit, (self.x, 8))
        if not wait:
            pygame.draw.line(
                screen,
                color=config.screen.white,
                start_pos=(self.x, config.pad.line_top),
                end_pos=(self.x, config.pad.line_bot),
                width=2,
            )
            screen.blit(self.sprite, self.sprite_pos)

    def pre_draw(self, screen):
        screen.blit(self.sprite, self._sprite_pos((1084, 185)))

    @property
    def sprite_pos(self):
        return self._sprite_pos((self.x, config.pad.top))

    def _sprite_pos(self, pos):
        x, y = pos
        w, h = self.sprite.get_size()
        a, b = config[self.n, "offset"]
        return x - w / 2 + a, y - h / 2 + b

    def set_x(self, x):
        #self.x = np.clip(x, self.left_lim, self.right_lim)
        if x < self.left_lim:
            self.x = self.right_lim - (self.left_lim - x)
        elif x > self.right_lim:
            self.x = self.left_lim + (x - self.right_lim)
        else:
            self.x = x
        return self.x

    def release(self, space):
        return Particle((self.x, config.pad.top), self.n, space)
