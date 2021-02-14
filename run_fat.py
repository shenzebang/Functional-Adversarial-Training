import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from tqdm import tqdm

from utils import load_data, Dx_cross_entropy, FunctionEnsemble, weak_oracle, Normalize

import numpy as np
import time

Dx_losses = {
    "logistic_regression": 123,
    "cross_entropy": Dx_cross_entropy
}
losses = {
    "logistic_regression": 123,
    "cross_entropy": lambda x, y: torch.nn.functional.cross_entropy(x, y, reduction='sum')
}


# todo: manage the gpu id
# todo: BUG when n_data mod n_workers is non-zero
norm = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
epsilon = 2./255


def fgd_step(f, data, label, step_size, oracle_n_steps, oracle_step_size, oracle_mb_size, init_weak_learner):
    f_data = f(data)
    target = Dx_loss(f_data, label)
    target = target.detach()
    g, _, _ = weak_oracle(target, data, oracle_step_size,
                          oracle_n_steps, init_weak_learner=init_weak_learner, mb_size=oracle_mb_size)
    f.add_function(g, -step_size)

def attack_step(model, data, label):
    delta = torch.zeros_like(data, requires_grad=True)
    opt = optim.SGD([delta], lr=5e-3)
    for t in range(100):
        pred = model(norm(data + delta))
        loss = -nn.CrossEntropyLoss()(pred, label)

        opt.zero_grad()
        loss.backward()
        opt.step()
        delta.data.clamp_(-epsilon, epsilon)

    return delta
if __name__ == '__main__':
    ts = time.time()
    algo = 'fat'
    parser = argparse.ArgumentParser(algo)

    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--weak_learner_hid_dims', type=str, default='32-32')
    parser.add_argument('--step_size_0', type=float, default=20.0)
    parser.add_argument('--loss', type=str, choices=['cross_entropy'], default='cross_entropy')
    parser.add_argument('--oracle_local_steps', type=int, default=1000)
    parser.add_argument('--fgd_n_steps', type=int, default=5)
    parser.add_argument('--oracle_step_size', type=float, default=0.001)
    parser.add_argument('--p', type=float, default=1, help='step size decay exponential')
    parser.add_argument('--oracle_mb_size', type=int, default=128)
    parser.add_argument('--n_global_rounds', type=int, default=100)
    parser.add_argument('--device', type=str, default="cuda")


    args = parser.parse_args()

    writer = SummaryWriter(
        f'out/{args.dataset}/{args.weak_learner_hid_dims}/rhog{args.step_size_0}_mb{args.oracle_mb_size}_p{args.p}_{algo}_{ts}'
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    hidden_size = tuple([int(a) for a in args.weak_learner_hid_dims.split("-")])

    # Load/split training data

    data, label, data_test, label_test, n_class, get_init_weak_learner = load_data(args, hidden_size, device)


    Dx_loss = Dx_losses[args.loss]
    loss = losses[args.loss]
    f_ens = FunctionEnsemble(device=device)
    delta = None

    for r in tqdm(range(args.n_global_rounds)):
        # fgd step
        for i in range(args.fgd_n_steps):
            step_size = args.step_size_0/(r*args.fgd_n_steps + i +1)
            fgd_step(f=f_ens,
                     data=data if delta is None else data+delta,
                     label=label,
                     step_size=step_size,
                     oracle_n_steps=args.oracle_n_steps,
                     oracle_step_size=args.oracle_step_size,
                     oracle_mb_size=args.oralce_mb_size,
                     init_weak_learner=get_init_weak_learner()
                     )
        # attack step
        delta = attack_step(model=f_ens,
                            data=data,
                            label=label
                            )

    print(args)

