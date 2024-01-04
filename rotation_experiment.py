import sys

import cv2
import numpy as np
import pygame
from PIL import Image

pygame.init()

screen_size = np.array([400, 400])

screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption("Testing Blit Rotation")
clock = pygame.time.Clock()

fruits = [
    "cherry",
    "strawberry",
    "grapes",
    "orange",
    "persimmon",
    "apple",
    "pear",
    "peach",
    "pineapple",
    "melon",
    "watermelon",
]

rsizes = [
    (40, 40),  # cherry
    (40, 43),  # strawberry
    (62, 56),  # grapes
    (72, 69),  # orange
    (88, 96),  # persimmon
    (112, 112),  # apple
    (130, 130),  # pear
    (156, 156),  # peach
    (175, 200),  # pineapple
    (250, 220),  # melon
    (250, 250),  # watermelon
]


def load_blit(index):
    out = pygame.image.load(f"blits/{fruits[index]}.png")
    out = pygame.transform.scale(out, rsizes[index])
    return out


def rplace(blit, ang, vect, radius):
    screen.fill((250, 250, 250))
    sprite = pygame.transform.rotate(blit.copy(), ang)
    sprite_size = np.array(sprite.get_size())

    rad = ang * np.pi / 180
    mat = np.array([
        [np.cos(rad), -np.sin(rad)],
        [np.sin(rad),  np.cos(rad)],
    ])

    pos = (screen_size - sprite_size) / 2 + (vect @ mat)

    pygame.draw.circle(screen, (0, 0, 0), screen_size // 2, radius)
    screen.blit(sprite, pos)


writer = cv2.VideoWriter("scenes/rotation_ex.mp4", cv2.VideoWriter.fourcc("a", "v", "c", "1"), 60, screen_size)
pygame.font.init()
scorefont = pygame.font.SysFont("monospace", 16)
index = 0
blit = load_blit(index)
ang = 0
vect = np.zeros(2)
radius = rsizes[index][0] // 2
rot = False

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print(vect, radius)
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                vect[0] -= 1
            elif event.key == pygame.K_RIGHT:
                vect[0] += 1
            elif event.key == pygame.K_UP:
                vect[1] -= 1
            elif event.key == pygame.K_DOWN:
                vect[1] += 1
            elif event.key == pygame.K_SPACE:
                rot = not rot
            elif event.key == pygame.K_RETURN:
                rot = False
                ang = 0
            elif event.key == pygame.K_PLUS:
                radius += 1
            elif event.key == pygame.K_MINUS:
                radius -= 1
            elif event.key == pygame.K_n:
                print(fruits[index], vect, radius)
                index = (index + 1) % 11
                blit = load_blit(index)
                ang = 0
                vect = np.zeros(2)
                radius = rsizes[index][0] // 2
                rot = False
            elif event.key == pygame.K_q:
                print(vect, radius)
                pygame.quit()
                sys.exit()
    if rot:
        ang = (ang + 5) % 360

    rplace(blit, ang, vect, radius)
    
    offset_label = scorefont.render(f"Offset: {vect}", 1, (0, 0, 0))
    radius_label = scorefont.render(f"Radius: {radius}", 1, (0, 0, 0))
    screen.blit(offset_label, (10, 350))
    screen.blit(radius_label, (10, 370))
    
    pygame.display.update()
    data = pygame.image.tostring(pygame.display.get_surface(), 'RGB')
    img = Image.frombytes('RGB', tuple(screen_size), data)
    writer.write(np.array(img)[..., ::-1])
    clock.tick(30)
