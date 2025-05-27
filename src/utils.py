import torch


device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.mps.is_available() else
                      "cpu")


def grad(u, x):
    return torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]


def array(x):
    return x.detach().cpu().numpy()


def tensor(x, **kwargs):
    return torch.tensor(x, dtype=torch.float32, device=device, **kwargs).reshape(-1, 1)


def lorenz_de(state, t, rho=28.0, sigma=10.0, beta=8.0/3.0):
    x, y, z = state
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z
