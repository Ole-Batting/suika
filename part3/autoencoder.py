import datetime

import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class Encoder(nn.Module):
    def __init__(self, n, act):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=n, out_channels=n, kernel_size=3, padding=1),
            act(),
            nn.Conv2d(in_channels=n, out_channels=n*2, kernel_size=3, padding=1),
            act(),
            nn.Conv2d(in_channels=n*2, out_channels=n*2, kernel_size=3, padding=1),
            act(),
            nn.Conv2d(in_channels=n*2, out_channels=n*2, kernel_size=2, stride=2, padding=0),
            act(),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, n, act):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=n*2, out_channels=n*2, kernel_size=2, stride=2, padding=0),
            act(),
            nn.Conv2d(in_channels=n*2, out_channels=n*2, kernel_size=3, padding=1),
            act(),
            nn.Conv2d(in_channels=n*2, out_channels=n, kernel_size=3, padding=1),
            act(),
            nn.Conv2d(in_channels=n, out_channels=n, kernel_size=3, padding=1),
            act(),
        )

    def forward(self, x):
        return self.net(x)


class ConvAE(nn.Module):
    def __init__(self, lr, di, nch, act):
        super().__init__()
        self.nch = nch
        self.act = act

        self.encoder = nn.ModuleList([Encoder(nch, act)])
        self.decoder = nn.ModuleList([Decoder(nch, act)])
        self.depth = 1

        self.criterion = nn.MSELoss()
        self.lr = lr
        self.di = di
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.laplace = torch.zeros((3, nch, 3, 3), dtype=torch.float32)
        self.laplace[0, 0] = lap.clone()
        self.laplace[1, 1] = lap.clone()
        self.laplace[2, 2] = lap.clone()

    def encode(self, x):
        for m in self.encoder:
            x = m(x)
        return x

    def decode(self, x):
        for m in self.decoder:
            x = m(x)
        if self.act == nn.Tanh:
            x = (x + 1) / 2
        else:
            x = nn.functional.sigmoid(x)
        return x

    def level_up(self):
        lvl = self.depth
        self.encoder.append(Encoder(self.nch * (2 ** lvl), self.act))
        self.decoder.insert(0, Decoder(self.nch * (2 ** lvl), self.act))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.depth += 1

    def set_lr(self, lr):
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        z = self.encode(x)
        y = self.decode(z)
        return y

    def both(self, x):
        z = self.encode(x)
        y = self.decode(z)
        return y, nn.functional.sigmoid(z)

    def laplacian(self, x):
        return nn.functional.conv2d(x, weight=self.laplace)

    def noise(self, x):
        noise = torch.randn_like(x) * 0.05
        return torch.clamp(x + noise, 0, 1)

    def fit(self, x, epochs):
        loss_rec = 0
        loss_lap = 0
        loss = 0

        for _ in range(epochs):
            y = self(x)
            loss_rec = self.criterion(x, y)

            lap_x = self.laplacian(x)
            lap_y = self.laplacian(y)
            loss_lap = self.criterion(lap_x, lap_y)

            #lat = self.encode(x)
            #noisy_lat = self.encode(self.noise(x))
            #loss_cl = self.criterion(lat, noisy_lat)

            loss = loss_rec + loss_lap #+ 0.2 * loss_cl

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.lr *= self.di
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return loss.item(), self.lr, loss_rec.item(), loss_lap.item()

    def spec(self, state_dim):
        d = self.depth
        s = 2
        c = self.nch
        sub_dim = state_dim // (s ** d)
        print("initial state size:", c * state_dim ** 2, (c, state_dim, state_dim))
        print("compressed size:", c * (2 ** d) * sub_dim ** 2, (c * 2 ** d, sub_dim, sub_dim))
        print(f"compression: 1:{(s ** (2 * d)) // (2 ** d)}")

    def save(self):
        torch.save(self.state_dict(), f'part3/output/AutoEncoder.pt')

    def load(self):
        self.load_state_dict(torch.load(f'part3/output/AutoEncoder.pt'))

def to_tensor(a):
    return torch.tensor(a, dtype=torch.float32)
