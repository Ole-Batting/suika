import copy
import math
import sys

import numpy as np
import pygame
import torch
from torch.distributions import Categorical

from config import config
from model import Pyramid


class PPO_discrete:
    def __init__(
            self,
            num_state_dims,
            num_action_dims,
            num_layers,
            lr,
            horizon,
            entropy_coef,
            entropy_decay,
            gamma,
            lambd,
            num_epochs,
            batch_size,
            clip_rate,
            l2_reg,
    ):
        self.num_state_dims = num_state_dims
        self.num_action_dims = num_action_dims
        self.num_layers = num_layers
        self.lr = lr
        self.horizon = horizon
        self.entropy_coef = entropy_coef
        self.entropy_decay = entropy_decay
        self.gamma = gamma
        self.lambd = lambd
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.clip_rate = clip_rate
        self.l2_reg = l2_reg

        self.actor = Pyramid(self.num_state_dims, self.num_action_dims, self.num_layers)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic = Pyramid(self.num_state_dims, 1, self.num_layers)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.state_buffer = np.zeros((self.horizon, self.num_state_dims), dtype=np.float32)
        self.action_buffer = np.zeros((self.horizon, 1), dtype=np.int64)
        self.reward_buffer = np.zeros((self.horizon, 1), dtype=np.float32)
        self.next_state_buffer = np.zeros((self.horizon, self.num_state_dims), dtype=np.float32)
        self.pi_action_buffer = np.zeros((self.horizon, 1), dtype=np.float32)
        self.done_buffer = np.zeros((self.horizon, 1), dtype=np.bool_)
        self.dw_buffer = np.zeros((self.horizon, 1), dtype=np.bool_)

        self.rng = np.random.default_rng(1)

    def select_action(self, state, deterministic=False):
        state = torch.from_numpy(state).float()
        with torch.no_grad():
            pi = self.actor.pi(state, softmax_dim=0)
            if deterministic:
                a = torch.argmax(pi).item()
                return a, None
            else:
                dist = Categorical(pi)
                action = dist.sample().item()
                pi_action = pi[action].item()
                return action, pi_action

    def train(self):
        self.entropy_coef *= self.entropy_decay
        state_tensor = torch.from_numpy(self.state_buffer)
        action_tensor = torch.from_numpy(self.action_buffer)
        reward_tensor = torch.from_numpy(self.reward_buffer)
        next_state_tensor = torch.from_numpy(self.next_state_buffer)
        old_pi_action_tensor = torch.from_numpy(self.pi_action_buffer)
        done_tensor = torch.from_numpy(self.done_buffer)

        with torch.no_grad():
            value_state = self.critic(state_tensor)
            value_next_state = self.critic(next_state_tensor)

            deltas = reward_tensor + self.gamma * value_next_state * (~done_tensor) - value_state
            deltas = deltas.flatten().numpy()
            adv = [0]

            for dlt, done in zip(deltas[::-1], done_tensor.flatten().numpy()):
                advantage = dlt + self.gamma * self.lambd * adv[-1] * (~done)
                adv.append(advantage)

            adv.reverse()
            adv = copy.deepcopy(adv[0:-1])
            adv = torch.tensor(adv).unsqueeze(1).float()
            td_target = adv + value_state

        # ppo update
        optim_iter_num = int(math.ceil(state_tensor.shape[0] / self.batch_size))

        for _ in range(self.num_epochs):
            perm = np.arange(state_tensor.shape[0])
            self.rng.shuffle(perm)
            perm = torch.LongTensor(perm)

            state_perm = state_tensor[perm].clone()
            action_perm = action_tensor[perm].clone()
            td_target_perm = td_target[perm].clone()
            adv_perm = adv[perm].clone()
            old_pi_action_perm = old_pi_action_tensor[perm].clone()

            # mini batch ppo update
            for i in range(optim_iter_num):
                index = slice(i * self.batch_size, min((i + 1) * self.batch_size, state_perm.shape[0]))

                # actor update
                prob = self.actor.pi(state_perm[index], softmax_dim=1)
                try:
                    entropy = Categorical(prob).entropy().sum(0, keepdim=True)
                except ValueError:
                    print(state_perm.shape, state_perm[index].shape)
                    raise
                prob_a = prob.gather(1, action_perm[index])
                ratio = torch.exp(torch.log(prob_a) - torch.log(old_pi_action_perm[index]))

                surr1 = ratio * adv_perm[index]
                surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv_perm[index]
                a_loss = -torch.min(surr1, surr2) - self.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                a_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor_optimizer.step()

                # critic update
                c_loss = (self.critic(state_perm[index]) - td_target_perm[index]).pow(2).mean()
                for name, param in self.critic.named_parameters():
                    if 'weight' in name:
                        c_loss += param.pow(2).sum() * self.l2_reg

                self.critic_optimizer.zero_grad()
                c_loss.backward()
                self.critic_optimizer.step()

    def put_data(self, s, a, r, n, p, d, idx):
        self.state_buffer[idx] = s
        self.action_buffer[idx] = a
        self.reward_buffer[idx] = r
        self.next_state_buffer[idx] = n
        self.pi_action_buffer[idx] = p
        self.done_buffer[idx] = d


def evaluate_policy(env, agent, turns=3, horizon=1024, screen=None, clock=None):
    total_scores = 0
    for j in range(turns):
        env.reset()
        s = env.state

        for i in range(horizon):  # for max
            # Take deterministic actions at test time
            go_random = agent.rng.random() < 0.02

            a, pi_a = agent.select_action(s, deterministic=go_random)
            env.step(a)
            s_next = env.state
            r = env.reward
            done = env.game_over

            total_scores += r
            s = s_next

            print(f"\revaluating {j:.0f} {i:.0f}: {total_scores:.0f} ", end='', flush=True)
            if done:
                break

            if screen is not None:
                if pygame.event.peek():
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()

                env.draw(screen)
                pygame.display.update()
                clock.tick(config.screen.fps)
        print('score', env.handler.data['score'])
    print()
    return int(total_scores/turns)
