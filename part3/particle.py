import numpy as np
import pygame
import pymunk

from config import config, CollisionTypes


class Particle(pymunk.Circle):
    def __init__(self, pos, n, space):
        self.n = n % 11
        super().__init__(
            body=pymunk.Body(body_type=pymunk.Body.DYNAMIC),
            radius=config[self.n, "radius"],
        )
        self.body.position = tuple(pos)
        self.density = config.physics.density
        self.elasticity = config.physics.elasticity
        self.collision_type = CollisionTypes.PARTICLE
        self.friction = config.physics.fruit_friction
        self.has_collided = False
        self.alive = True
        space.add(self.body, self)

    def draw(self, screen):
        if self.alive:
            sprite = pygame.transform.rotate(
                config[self.n, "blit"].copy(),
                -self.body.angle * 180/np.pi,
            )
            screen.blit(sprite, self.sprite_pos(sprite))

    def kill(self, space):
        space.remove(self.body, self)
        self.alive = False

    @property
    def pos(self):
        return np.array(self.body.position)

    def sprite_pos(self, sprite):
        x, y = self.body.position
        w, h = sprite.get_size()
        a, b = self.sprite_offset
        return x - w / 2 + a, y - h / 2 + b

    @property
    def sprite_offset(self):
        ang = self.body.angle
        mat = np.array([
            [np.cos(ang), -np.sin(ang)],
            [np.sin(ang), np.cos(ang)],
        ])
        arr = np.array(config[self.n, "offset"])
        return mat @ arr
