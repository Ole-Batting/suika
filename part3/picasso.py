import datetime
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pygame
import torch
from einops import rearrange

from autoencoder import ConvAE, to_tensor
from config import config
from environment import Suika

torch.set_flush_denormal(True)

n_frame_skip = 8
n_frame_skip2 = 8
view_dim = 256
nxp = 128
#lvd = [(2, 4), (4, 4), (4, 8), (8, 8), (8, 16), (16, 16)]
lvd = [(2, 3), (3, 4), (4, 6), (6, 8), (8, 12), (12, 16), (16, 24)]

state_dim = 64
slope = 2
nch = 3
act = torch.nn.Tanh

batch_sub = 12
batch_size = batch_sub * batch_sub
epochs = 6
lr = 3e-3
di = 0.98
tol = 0.01
ninc = 50

rng = np.random.default_rng(1)
screen = pygame.display.set_mode((config.screen.width, config.screen.height))
pygame.display.set_caption("PySuika")
clock = pygame.time.Clock()

env = Suika(tag=1, n_frame_skip=n_frame_skip)
lae = ConvAE(lr=lr, di=di, nch=nch, act=act)
lae.spec(state_dim)

batch = torch.zeros((batch_size, nch, state_dim, state_dim))

writers = {}

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line1, line2, line3, = ax.plot(np.zeros((2, 3)), label=['total', 'rec', 'lap'])
plt.legend()

loss_log = []
loss_rec_log = []
loss_lap_log = []


def show_and_tell(window_name, frame):
    if window_name not in writers.keys():
        wri = cv2.VideoWriter(
            f"part3/output/{window_name}.mp4",
            cv2.VideoWriter.fourcc("a", "v", "c", "1"),
            60,
            (frame.shape[1], frame.shape[0]),
        )
        writers[window_name] = wri
    if frame.dtype is not np.uint8:
        frame = (frame * 255).astype(np.uint8)

    if frame.ndim == 2 or frame.shape[2] == 1:
        frame = cv2.applyColorMap(frame, cv2.COLORMAP_MAGMA)

    writers[window_name].write(frame)
    cv2.imshow(window_name, frame)


for epoch in range(1000):
    for j in range(batch_size // 2):
        for _ in range(n_frame_skip2):
            if pygame.event.peek():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        lae.save()
                        for wri in writers.values():
                            wri.release()
                        cv2.destroyAllWindows()
                        sys.exit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_u:
                            lae.level_up()
                            lae.spec(state_dim)
                        elif event.key == pygame.K_r:
                            lae.set_lr(lr)
                        elif event.key == pygame.K_s:
                            lae.save()
                        elif event.key == pygame.K_s:
                            lae.load()
                        elif event.key == pygame.K_UP:
                            lae.lr *= 1.2
                        elif event.key == pygame.K_DOWN:
                            lae.lr /= 1.2

            if env.game_over:
                env.reset()

            pi_action = rng.random(4)
            action = np.argmax(pi_action)
            env.step(action)

            #env.draw(screen)
            #pygame.display.update()
            #clock.tick(config.screen.fps)

        state = cv2.resize(env.state, (state_dim, state_dim))
        batch[rng.integers(0, batch_size)] = rearrange(to_tensor(state), "h w c -> c h w")

        with torch.no_grad():
            input_tensor = rearrange(to_tensor(state), "h w (b c) -> b c h w", b=1)

            out, lat = lae.both(input_tensor)
            output = rearrange(out, "b c h w -> h w (b c)").numpy()

            batch_im = rearrange(batch, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=batch_sub, b2=batch_sub).numpy()

            rows, cols = lvd[lae.depth - 1]
            lat = rearrange(lat, "b (c1 c2) h w -> (c1 h) (c2 w) b", c1=rows, c2=cols).numpy()

            #noise = lae.noise(input_tensor)
            #noisy_lat = torch.nn.functional.sigmoid(lae.encode(noise))
            #noise = rearrange(noise, "b c h w -> h w (b c)").numpy()
            #noisy_lat = rearrange(noisy_lat, "b (c1 c2) h w -> (c1 h) (c2 w) b", c1=rows, c2=cols).numpy()

            lap_x = torch.nn.functional.sigmoid(lae.laplacian(input_tensor))
            lap_x = rearrange(lap_x, "b c h w -> h w (b c)").numpy()

            lap_y = torch.nn.functional.sigmoid(lae.laplacian(out))
            lap_y = rearrange(lap_y, "b c h w -> h w (b c)").numpy()

            show_and_tell('raw', env.state[..., [2, 1, 0]])
            show_and_tell('lap_x', cv2.resize(lap_x, (view_dim, view_dim)))
            show_and_tell('lap_y', cv2.resize(lap_y, (view_dim, view_dim)))
            #show_and_tell('noisy_lat', cv2.resize(noisy_lat, dsize=(view_dim, view_dim), interpolation=cv2.INTER_NEAREST))
            #show_and_tell('noise', cv2.resize(noise[..., [2, 1, 0]], (view_dim, view_dim)))
            show_and_tell('reconstruction', cv2.resize(output[..., [2, 1, 0]], (view_dim, view_dim)))
            show_and_tell('latent', cv2.resize(lat, dsize=(view_dim, view_dim), interpolation=cv2.INTER_NEAREST))
            show_and_tell('state', cv2.resize(state[..., [2, 1, 0]], (view_dim, view_dim)))
            #show_and_tell('batch', batch_im[..., [2, 1, 0]])


    loss, _lr, loss_rec, loss_lap = lae.fit(batch, epochs)

    loss_log.append(loss)
    loss_rec_log.append(loss_rec)
    loss_lap_log.append(loss_lap)

    if loss_log[-ninc] - loss_log[-1] < tol:
        lae.level_up()
        lae.spec(state_dim)

    xdata = np.arange(len(loss_log))

    line1.set_xdata(xdata)
    line1.set_ydata(loss_log)
    line2.set_xdata(xdata)
    line2.set_ydata(loss_rec_log)
    line3.set_xdata(xdata)
    line3.set_ydata(loss_lap_log)

    ax.set_xlim(0, len(loss_log))
    ax.set_ylim(0, 1.2 * max(loss_log))

    fig.canvas.draw()
    fig.canvas.flush_events()

    print(f"{epoch}: {loss}, {_lr}")


