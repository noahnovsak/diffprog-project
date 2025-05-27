import torch
import torch.nn as nn

from model import archs
from utils import grad, device


class LorenzAttractor(nn.Module):
    def __init__(self, ics, t0, t1, n_t=256, rho=28.0, sigma=10.0, beta=8.0/3.0,
                 causal=True, mlp='ModifiedMLP', **kwargs):
        super().__init__()

        self.ics = torch.tensor(ics, device=device)
        self.rho = rho
        self.sigma = sigma
        self.beta = beta

        self.t0 = t0
        self.t1 = t1
        self.t = torch.linspace(t0, t1, n_t, requires_grad=True, device=device).reshape(-1, 1)

        self.causal = causal
        if causal:
            self.tol = 1e-4
            self.M = torch.tril(torch.ones((n_t, n_t), device=device), diagonal=1)

        self.net = archs[mlp](**kwargs).to(device)
        self.net.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, t):
        return self.net(t) *  t + self.ics

    def residuals(self, t):
        x, y, z = self.forward(t).split(1, dim=1)
        x_t = grad(x, t)
        y_t = grad(y, t)
        z_t = grad(z, t)

        res_x = x_t - self.sigma * (y - x)
        res_y = y_t - x * (self.rho - z) + y
        res_z = z_t - x * y + self.beta * z

        return res_x, res_y, res_z

    def causal_w(self, r1, r2, r3):
        if not self.causal:
            return 1
        with torch.no_grad():
            w = torch.exp(- self.tol * self.M @ (r1**2 + r2**2 + r3**2))
        return w

    def loss(self, t):
        r1, r2, r3 = self.residuals(t)
        w = self.causal_w(r1, r2, r3)
        return torch.mean(w * (r1**2 + r2**2 + r3**2))
