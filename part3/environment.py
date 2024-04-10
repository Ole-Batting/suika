import numpy as np
import pygame
import pymunk

from cloud import Cloud
from collision import collide
from config import config, CollisionTypes
from particle import Particle
from text import score, gameover
from wall import Wall


class Suika:
    def __init__(self, tag='', n_frame_skip=1):

        # Spec
        self.n_frame_skip = n_frame_skip
        self.width = config.screen.width
        self.height = config.screen.height
        self.num_action_dims = 4

        # Hyper
        self.reward_for_score = 1
        self.reward_for_idle = -0.2
        self.reward_for_impatience = 0
        self.reward_for_game_over = -100

        self.reset()

        self.loc_arr = None
        self.num_state_dims = self.state.shape

    def reset(self):

        # Physics
        self.space = pymunk.Space()
        self.space.gravity = (0, config.physics.gravity)
        self.space.damping = config.physics.damping
        self.space.collision_bias = config.physics.bias

        # Walls and cloud
        self.left = Wall(config.top_left, config.bot_left, self.space)
        self.bottom = Wall(config.bot_left, config.bot_right, self.space)
        self.right = Wall(config.bot_right, config.top_right, self.space)
        self.walls = [self.left, self.bottom, self.right]
        self.cloud = Cloud()

        # Game state
        self.wait_for_next = 0
        self.idle_count = 0
        self.impatient = False
        self.game_over = False

        # Collision Handler
        self.handler = self.space.add_collision_handler(CollisionTypes.PARTICLE, CollisionTypes.PARTICLE)
        self.handler.begin = collide
        self.handler.data["score"] = 0
        self.last_score = 0

    def _step(self):
        if self.wait_for_next > 1:
            self.wait_for_next -= 1
        if self.wait_for_next == 1:
            self.cloud.step()
            self.wait_for_next -= 1

        for p in self.space.shapes:
            if not isinstance(p, Particle):
                continue
            if p.pos[1] < config.pad.killy and p.has_collided:
                self.game_over = True

        # Step game
        self.space.step(1 / config.screen.fps)

    def step(self, action):
        if not self.game_over:
            self.impatient = False
            next_idle_count = 0

            if action == 0:
                next_idle_count = self.idle_count + 1
            elif action == 1:
                self.cloud.move_left(self.n_frame_skip)
            elif action == 2:
                self.cloud.move_left(self.n_frame_skip)
            elif action == 3:
                if self.wait_for_next == 0:
                    self.cloud.release(self.space)
                    self.wait_for_next = config.screen.delay
                else:
                    self.impatient = True

            self.idle_count = next_idle_count

        for _ in range(self.n_frame_skip):
            self._step()
            if self.game_over:
                break

    def draw(self, surface: pygame.Surface):
        surface.blit(config.background_blit, dest=(0, 0))
        self.cloud.draw(surface, self.wait_for_next)
        for p in self.space.shapes:
            if not isinstance(p, Particle):
                continue
            p.draw(surface)
        score(self.handler.data['score'], surface)
        if self.game_over:
            gameover(surface)

    @property
    def state(self):
        surf = pygame.Surface((config.screen.width, config.screen.height))
        self.draw(surf)
        image = np.transpose(pygame.surfarray.array3d(surf), axes=(1, 0, 2))
        image = image[:, config.pad.left:config.pad.right] / 255

        if self.loc_arr is None and False:
            h, w = image.shape[:2]
            self.loc_arr = np.arange(h * w, dtype=float).reshape(h, w, 1)
            self.loc_arr -= h * w / 2
            self.loc_arr /= h * w / 2
        # image = np.dstack((image, self.loc_arr))
        return image

    @property
    def reward(self):
        score = self.handler.data['score']
        score_change = score - self.last_score

        out = score_change * self.reward_for_score
        out += min(0, self.idle_count - 5) * self.reward_for_idle

        if self.impatient:
            out += self.reward_for_impatience

        self.last_score = score
        if self.game_over:
            out += self.reward_for_game_over
        return out

