import numpy as np
import torch

from scipy.integrate import odeint
from tqdm.notebook import trange

from utils import array, tensor, lorenz_de


def eval(model):
     with torch.no_grad():
        t = np.arange(model.t0, model.t1, 0.01)
        u_test = odeint(lorenz_de, array(model.ics), t)
        u_pred = array(model(tensor(t)))
        rel_l2 = np.linalg.norm(u_test - u_pred) / np.linalg.norm(u_test)
        return f" Rel L2: {rel_l2:.5f}"


def train(model, epochs=50000):
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    sched = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=.9)

    log_rate = epochs // 50
    update_rate = log_rate * 5
    eval_rate = update_rate * 2

    for i in trange(epochs + 1):
        optim.zero_grad()
        loss = model.loss(model.t)
        loss.backward()
        optim.step()

        log = f"Epoch: {i:>6} Loss: {loss.item():>.2e}"

        if i % update_rate == 0:
            sched.step()
            log += f" LR: {sched.get_last_lr()[0]:.1e}"

            if model.causal:
                if getattr(model, 'w_', None) is None:
                    model.w_ = []
                model.w_.append(model.causal_w(*model.residuals(model.t)))
                w_min = model.w_[-1].min()
                log += f" Min w: {w_min:.5f}"

                if w_min > 0.98:
                    model.tol *= np.sqrt(10)
                    log += f" Tol: {model.tol:.1e}"

        if i % eval_rate == 0:
           log += eval(model)

        if i % log_rate == 0:
            print(log)
