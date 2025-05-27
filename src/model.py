import torch
import torch.nn as nn


class FourierEmbedding(nn.Module):
    def __init__(self, embed_dim=256, input_dim=1, scale=1):
        super().__init__()
        self.kernel = nn.Parameter(torch.randn((input_dim, embed_dim // 2)) * scale)

    def forward(self, x):
        prod = x @ self.kernel
        return torch.hstack([torch.cos(prod), torch.sin(prod)])


class MLP(nn.Module):
    def __init__(self, layers=[1, 256, 256, 3], activation=nn.Tanh, embed=None):
        super().__init__()

        self.embed = embed is not None
        if self.embed:
            self.embedding = FourierEmbedding(**embed)

        net = []
        for i in range(len(layers) - 1):
            net.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                net.append(activation())
        self.net = nn.Sequential(*net)

    def forward(self, x):
        if self.embed:
            x = self.embedding(x)
        return self.net(x)


class ModifiedMLP(nn.Module):
    def __init__(self, layers=[1, 256, 256, 3], activation=nn.Tanh, embed=None):
        super().__init__()

        self.embed = embed is not None
        if self.embed:
            self.embedding = FourierEmbedding(**embed)

        self.u = nn.Sequential(
            nn.Linear(layers[1], layers[1]),
            activation(),
        )
        self.v = nn.Sequential(
            nn.Linear(layers[1], layers[1]),
            activation(),
        )

        self.n_layers = len(layers)
        self.net = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.net.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                self.net.append(activation())

    def forward(self, x):
        if self.embed:
            x = self.embedding(x)
        for i in range(0, self.n_layers, 2):
            x = self.net[i](x)
            x = self.net[i + 1](x)
            x = x * self.u(x) + (1 - x) * self.v(x)
        x = self.net[-1](x)
        return x


archs = {
    'MLP': MLP,
    'ModifiedMLP': ModifiedMLP,
}
