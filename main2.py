import sys

import numpy as np
import pygame
import pymunk

pygame.init()
rng = np.random.default_rng()

# blits
BG = pygame.image.load("blits/suikabackground.png")
CLOUD = pygame.image.load("blits/cloudboi.png")

# Constants
SIZE = WIDTH, HEIGHT = np.array([1280, 720])
PAD = (415, 85, 863, 675)
W_COLOR = (170, 100, 0)
COLORS = [
    (245, 0, 0),
    (250, 100, 100),
    (150, 20, 250),
    (250, 210, 10),
    (250, 150, 0),
    (245, 0, 0),
    (250, 250, 100),
    (255, 180, 180),
    (255, 255, 0),
    (100, 235, 10),
    (0, 185, 0),
]
FPS = 120
RADII = [17, 25, 32, 38, 50, 63, 75, 87, 100, 115, 135]
DENSITY = 0.001
ELASTICITY = 0.1
IMPULSE = 10000
GRAVITY = 2000
DAMPING = 0.8
NEXT_DELAY = FPS
BIAS = 0.00001
POINTS = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66]


class Particle(pymunk.Circle):
    def __init__(self, pos, n, space):
        self.n = n % 11
        super().__init__(
            body=pymunk.Body(body_type=pymunk.Body.DYNAMIC),
            radius=RADII[self.n]
        )
        self.body.position = tuple(pos)
        self.density = DENSITY
        self.elasticity = ELASTICITY
        self.collision_type = 1
        self.friction = 0.2
        self.has_collided = False
        self.alive = True
        space.add(self.body, self)

    def draw(self, screen):
        if self.alive:
            c1 = np.array(COLORS[self.n])
            c2 = (c1 * 0.8).astype(int)
            pygame.draw.circle(screen, tuple(c2), self.body.position, self.radius)
            pygame.draw.circle(screen, tuple(c1), self.body.position, self.radius * 0.9)

    def kill(self, space):
        space.remove(self.body, self)
        self.alive = False

    @property
    def pos(self):
        return np.array(self.body.position)


class PreParticle:
    def __init__(self, x, n):
        self.n = n % 11
        self.radius = RADII[self.n]
        self.x = x
        print(f"PreParticle {id(self)} created")

    def draw(self, screen, wait):
        c1 = np.array(COLORS[self.n])
        c2 = (c1 * 0.8).astype(int)
        screen.blit(CLOUD, (self.x, 8))
        if not wait:
            pygame.draw.circle(screen, tuple(c2), (self.x, PAD[1]), self.radius)
            pygame.draw.circle(screen, tuple(c1), (self.x, PAD[1]), self.radius * 0.9)

    def set_x(self, x):
        lim = PAD[0] + self.radius
        self.x = np.clip(x, lim, WIDTH - lim)

    def release(self, space):
        return Particle((self.x, PAD[1]), self.n, space)


class Wall(pymunk.Segment):
    def __init__(self, a, b, space):
        super().__init__(body=pymunk.Body(body_type=pymunk.Body.STATIC), a=a, b=b, radius=2)
        self.friction = 10
        space.add(self.body, self)


def resolve_collision(p1, p2, space, particles):
    distance = np.linalg.norm(p1.pos - p2.pos)
    if distance < 2 * p1.radius:
        p1.kill(space)
        p2.kill(space)
        pn = Particle(np.mean([p1.pos, p2.pos], axis=0), p1.n+1, space)
        for p in particles:
            if p.alive:
                vector = p.pos - pn.pos
                distance = np.linalg.norm(vector)
                if distance < pn.radius + p.radius:
                    impulse = IMPULSE * vector / (distance ** 2)
                    p.body.apply_impulse_at_local_point(tuple(impulse))
        return pn
    return None


# Create Pygame window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("PySuika")
clock = pygame.time.Clock()
pygame.font.init()
scorefont = pygame.font.SysFont("monospace", 32)
overfont = pygame.font.SysFont("monospace", 72)

space = pymunk.Space()
space.gravity = (0, GRAVITY)
space.damping = DAMPING
space.collision_bias = BIAS

# Walls
left = Wall((PAD[0], PAD[1]), (PAD[0], PAD[3]), space)
bottom = Wall((PAD[0], PAD[3]), (PAD[2], PAD[3]), space)
right = Wall((PAD[2], PAD[3]), (PAD[2], PAD[1]), space)
walls = [left, bottom, right]


# List to store particles
wait_for_next = 0
next_particle = PreParticle(WIDTH//2, rng.integers(0, 5))
particles = []

# Collision Handler
handler = space.add_collision_handler(1, 1)


def collide(arbiter, space, data):
    pa1, pa2 = arbiter.shapes
    cond = bool(pa1.n != pa2.n)
    pa1.has_collided = cond
    pa2.has_collided = cond
    if not cond:
        new_particle = resolve_collision(pa1, pa2, space, data["particles"])
        data["particles"].append(new_particle)
        data["score"] += POINTS[pa1.n]
    return cond


handler.begin = collide
handler.data["particles"] = particles
handler.data["score"] = 0

# Main game loop
game_over = False

while not game_over:
    for event in pygame.event.get():
        if any([
                event.type == pygame.QUIT,
                event.type == pygame.KEYDOWN and event.key in [pygame.K_q, pygame.K_ESCAPE],
        ]):
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN and wait_for_next == 0:
            particles.append(next_particle.release(space))
            wait_for_next = NEXT_DELAY

    next_particle.set_x(pygame.mouse.get_pos()[0])

    if wait_for_next > 1:
        wait_for_next -= 1
    elif wait_for_next == 1:
        next_particle = PreParticle(next_particle.x, rng.integers(0, 5))
        wait_for_next -= 1

    # Draw background and particles
    screen.blit(BG, (0, 0))
    next_particle.draw(screen, wait_for_next)
    for p in particles:
        p.draw(screen)
        if p.pos[1] < 160 and p.has_collided:
            label = overfont.render("Game Over!", 1, (0, 0, 0))
            screen.blit(label, (30, 160))
            game_over = True
    label = scorefont.render(f"Score: {handler.data['score']}", 1, (0, 0, 0))
    screen.blit(label, (10, 10))

    space.step(1/FPS)
    pygame.display.update()
    clock.tick(FPS)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_RETURN, pygame.K_SPACE, pygame.K_q, pygame.K_ESCAPE]:
                pygame.quit()
                sys.exit()
