import sys
from datetime import datetime as dt

import pygame

from config import config
from environment import Suika
from ppo_discrete import PPO_discrete, evaluate_policy

n_frame_skip = 8
n_sub_sample = 8
headless = True

screen = None
clock = None
if not headless:
    screen = pygame.display.set_mode((config.screen.width, config.screen.height))
    pygame.display.set_caption("PySuika")
    clock = pygame.time.Clock()

env = Suika(n_frame_skip=n_frame_skip, n_sub_sample=n_sub_sample)
eval_env = Suika(n_frame_skip=n_frame_skip, n_sub_sample=n_sub_sample)
agent = PPO_discrete(
    num_state_dims=env.num_state_dims,
    num_action_dims=env.num_action_dims,
    num_layers=2,
    lr=1e-5,
    horizon=256,
    entropy_coef=0,
    entropy_decay=0.99,
    gamma=0.99,
    lambd=0.95,
    num_epochs=10,
    batch_size=64,
    clip_rate=0.2,
    l2_reg=0,
)

max_horizons = 100
train_horizons = 5
test_horizons = 2
buffer_length = 0
total_steps = 0
score = 0
fps = 0

while total_steps < (agent.horizon * max_horizons):
    env.reset()
    state = env.state
    done = False
    dt0 = dt.now()
    fr0 = total_steps

    while not done:
        action, pi_action = agent.select_action(state)
        env.step(action)
        next_state = env.state
        reward = env.reward
        done = env.game_over

        agent.put_data(state, action, reward, next_state, pi_action, done, idx=buffer_length)
        state = next_state

        buffer_length += 1
        total_steps += 1

        if buffer_length == agent.horizon:
            agent.train()
            buffer_length = 0
            dt1 = dt.now()
            fr1 = total_steps
            fps = n_frame_skip * (fr1 - fr0) / (dt1 - dt0).total_seconds()
            dt0 = dt1
            fr0 = fr1

        if total_steps % (train_horizons * agent.horizon) == 0:
            pygame.display.set_caption("PySuika eval")
            score = evaluate_policy(
                eval_env, agent, turns=test_horizons, horizon=agent.horizon, screen=screen, clock=clock
            )
            pygame.display.set_caption("PySuika train")
            env.reset()

        print(f"\r{total_steps:.0f}: {score:.0f} ({fps:.0f} fps)", end='', flush=True)

        if not headless:
            if pygame.event.peek():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

            env.draw(screen)
            pygame.display.update()
            clock.tick(config.screen.fps)




