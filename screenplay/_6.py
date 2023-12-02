# !!animate
from _5 import *  # !!ignore
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
pad = 20
left = Wall(A, B, space)
bottom = Wall(B, C, space)
right = Wall(C, D, space)
walls = [left, bottom, right]

# List to store particles
wait_for_next = 0
next_particle = PreParticle(WIDTH//2, rng.integers(0, 5))
particles = []

game_over = False
