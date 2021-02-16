import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from tqdm import tqdm

from utils import load_data, Dx_cross_entropy, FunctionEnsemble, weak_oracle, Normalize, chunks

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


def sgd_step(f, data, label, sgd_n_steps, sgd_step_size, sgd_mb_size):
    opt = optim.SGD(f.parameters(), lr=sgd_step_size)
    for i in range(sgd_n_steps):
        opt.zero_grad()
        index = torch.randint(data.shape[0], [sgd_mb_size])
        loss_f = loss(f(data[index]), label[index])
        loss_f.backward()
        opt.step()

def attack_step(model, data, label, mb_size=128):
    delta = torch.zeros_like(data, requires_grad=True)
    opt = optim.SGD([delta], lr=5e-3)

    # chunk the data due to GPU memory limitation
    full_index = range(data.shape[0])
    index_list = chunks(full_index, mb_size)
    for index in index_list:
        for t in range(100):
            pred = model(norm(data[index] + delta[index]))
            loss = -nn.CrossEntropyLoss()(pred, label[index])

            opt.zero_grad()
            loss.backward()
            opt.step()
            delta[index].data.clamp_(-epsilon, epsilon)

    return delta
if __name__ == '__main__':
    ts = time.time()
    algo = 'at'
    parser = argparse.ArgumentParser(algo)

    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--weak_learner_hid_dims', type=str, default='32-32')
    parser.add_argument('--sgd_step_size', type=float, default=0.005)
    parser.add_argument('--loss', type=str, choices=['cross_entropy'], default='cross_entropy')
    parser.add_argument('--sgd_n_steps', type=int, default=100)
    parser.add_argument('--oracle_step_size', type=float, default=0.001)
    parser.add_argument('--p', type=float, default=1, help='step size decay exponential')
    parser.add_argument('--sgd_mb_size', type=int, default=128)
    parser.add_argument('--n_global_rounds', type=int, default=100)
    parser.add_argument('--device', type=str, default="cuda")


    args = parser.parse_args()

    writer = SummaryWriter(
        f'out/{args.dataset}/{args.weak_learner_hid_dims}/rhog{args.sgd_step_size}_mb{args.sgd_mb_size}_p{args.p}_{algo}_{ts}'
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    hidden_size = tuple([int(a) for a in args.weak_learner_hid_dims.split("-")])

    # Load/split training data

    data, label, data_test, label_test, n_class, get_init_weak_learner = load_data(args, hidden_size, device)


    Dx_loss = Dx_losses[args.loss]
    loss = losses[args.loss]
    f = get_init_weak_learner()
    f.requires_grad_(True)
    delta = None

    for r in tqdm(range(args.n_global_rounds)):
        # sgd step
        sgd_step(f=f,
                 data=data if delta is None else (data+delta).detach(),
                 label=label,
                 sgd_step_size=args.sgd_step_size,
                 sgd_n_steps=args.sgd_n_steps,
                 sgd_mb_size=args.sgd_mb_size
                 )
        # attack step
        delta = attack_step(model=f,
                            data=data,
                            label=label
                            )
        # test on natural data
        f_data_test = f(data_test)
        loss_round = loss(f_data_test, label_test)
        writer.add_scalar(
            f"global loss vs round, {args.dataset}/natural_test",
            loss_round, r)
        pred = f_data_test.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = np.true_divide(pred.eq(label_test.view_as(pred)).sum().item(), label_test.shape[0])
        writer.add_scalar(
            f"correct rate vs round, {args.dataset}/natural_test",
            correct, r)

        # test on adversarial data
        delta_test = attack_step(model=f,
                            data=data_test,
                            label=label_test
                            )
        f_data_test = f(data_test+delta_test)
        loss_round = loss(f_data_test, label_test)
        writer.add_scalar(
            f"global loss vs round, {args.dataset}/adv_test",
            loss_round, r)
        pred = f_data_test.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct_adv = np.true_divide(pred.eq(label_test.view_as(pred)).sum().item(), label_test.shape[0])
        writer.add_scalar(
            f"correct rate vs round, {args.dataset}/adv_test",
            correct_adv, r)

        print(f"Round {r}, {correct} on natural data, {correct_adv} on adv data")
    print(args)

