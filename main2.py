import sys

import numpy as np
import pygame
import pymunk

from config import Config

pygame.init()
rng = np.random.default_rng()

# Config
config = Config()


class Particle(pymunk.Circle):
    def __init__(self, pos, n, space):
        self.n = n % 11
        super().__init__(body=pymunk.Body(body_type=pymunk.Body.DYNAMIC), radius=config[self.n, "radius"])
        self.body.position = tuple(pos)
        self.density = config.physics.density
        self.elasticity = config.physics.elasticity
        self.collision_type = 1
        self.friction = 0.4
        self.has_collided = False
        self.alive = True
        space.add(self.body, self)

    def draw(self, screen):
        if self.alive:
            sprite = pygame.transform.rotate(config[self.n, "blit"].copy(), -self.body.angle * 180/np.pi)
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


class PreParticle:
    def __init__(self):
        self.x = config.screen.width // 2
        self.n = rng.integers(0, 5)
        self.radius = config[self.n, "radius"]
        self.sprite = config[self.n, "blit"]

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
        left_lim = config.pad.left + self.radius
        right_lim = config.pad.right - self.radius
        self.x = np.clip(x, left_lim, right_lim)

    def release(self, space):
        return Particle((self.x, config.pad.top), self.n, space)


class Cloud:
    def __init__(self):
        self.curr = PreParticle()
        self.next = PreParticle()

    def draw(self, screen, wait):
        self.curr.draw(screen, wait)
        self.next.pre_draw(screen)

    def release(self, space):
        return self.curr.release(space)

    def step(self):
        self.curr = self.next
        self.next = PreParticle()


class Wall(pymunk.Segment):
    def __init__(self, a, b, space):
        super().__init__(body=pymunk.Body(body_type=pymunk.Body.STATIC), a=a, b=b, radius=2)
        self.friction = 10
        space.add(self.body, self)


def resolve_collision(particle1, particle2, space, particles):
    distance = np.linalg.norm(particle1.pos - particle2.pos)
    if distance < 2 * particle1.radius:
        particle1.kill(space)
        particle2.kill(space)
        new_particle = Particle(np.mean([particle1.pos, particle2.pos], axis=0), particle1.n + 1, space)
        for p in particles:
            if p.alive:
                vector = p.pos - new_particle.pos
                distance = np.linalg.norm(vector)
                if distance < new_particle.radius + p.radius:
                    impulse = config.physics.impulse * vector / (distance ** 2)
                    p.body.apply_impulse_at_local_point(tuple(impulse))
        return new_particle
    return None


# Create Pygame window
screen = pygame.display.set_mode((config.screen.width, config.screen.height))
pygame.display.set_caption("PySuika")
clock = pygame.time.Clock()
pygame.font.init()
scorefont = pygame.font.SysFont("monospace", 32)
overfont = pygame.font.SysFont("monospace", 72)

space = pymunk.Space()
space.gravity = (0, config.physics.gravity)
space.damping = config.physics.damping
space.collision_bias = config.physics.bias

# Walls
left = Wall(config.top_left, config.bot_left, space)
bottom = Wall(config.bot_left, config.bot_right, space)
right = Wall(config.bot_right, config.top_right, space)
walls = [left, bottom, right]


# List to store particles
wait_for_next = 0
cloud = Cloud()
particles = []

# Collision Handler
handler = space.add_collision_handler(1, 1)


def collide(arbiter, space, data):
    particle1, particle2 = arbiter.shapes
    alive = particle1.alive and particle2.alive
    same = particle1.n == particle2.n
    particle1.has_collided = not same
    particle2.has_collided = not same
    if same and alive:
        new_particle = resolve_collision(particle1, particle2, space, data["particles"])
        data["particles"].append(new_particle)
        data["score"] += config[particle1.n, "points"]
    return not same and alive


handler.begin = collide
handler.data["particles"] = particles
handler.data["score"] = 0

# Main game loop
game_over = False
go_ham = False

while not game_over:
    # Take user input
    if pygame.event.peek():
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN and wait_for_next == 0:
                particles.append(cloud.release(space))
                wait_for_next = config.screen.delay

    cloud.curr.set_x(pygame.mouse.get_pos()[0])

    # Delay between fruit drop
    if wait_for_next > 1:
        wait_for_next -= 1
    if wait_for_next == 1:
        cloud.step()
        wait_for_next -= 1

    if go_ham and wait_for_next == 0:
        particles.append(cloud.release(space))
        wait_for_next = config.screen.delay

    # Draw background and particles
    screen.blit(config.background_blit, (0, 0))
    cloud.draw(screen, wait_for_next)
    for p in particles:
        p.draw(screen)
        if p.pos[1] < config.pad.killy and p.has_collided:
            label = overfont.render("Game Over!", 1, (0, 0, 0))
            screen.blit(label, (30, 160))
            game_over = True
    label = scorefont.render(f"Score: {handler.data['score']}", 1, (0, 0, 0))
    screen.blit(label, (10, 10))

    # Time step
    space.step(1/config.screen.fps)
    pygame.display.update()
    clock.tick(config.screen.fps)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_RETURN, pygame.K_SPACE, pygame.K_q, pygame.K_ESCAPE]:
                pygame.quit()
                sys.exit()
