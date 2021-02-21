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
    "cross_entropy": lambda x, y: torch.nn.functional.cross_entropy(x, y)
}


# todo: manage the gpu id
# todo: BUG when n_data mod n_workers is non-zero
norm = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


def fgd_step(f, data, label, step_size, oracle_n_steps, oracle_step_size, oracle_mb_size, init_weak_learner):
    f_data = f(data)
    target = Dx_loss(f_data, label)
    target = target.detach()
    g, _, _ = weak_oracle(target, data, oracle_step_size,
                          oracle_n_steps, init_weak_learner=init_weak_learner, mb_size=oracle_mb_size)
    f.add_function(g, -step_size)

def attack_step(model, data, label, epsilon, attack_lr=5e-3, mb_size=128):
    delta = torch.zeros_like(data, requires_grad=True)

    if epsilon > 0:
        opt = optim.SGD([delta], lr=attack_lr)

        if mb_size > 0:
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
        else:
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
    parser.add_argument('--oracle_n_steps', type=int, default=1000)
    parser.add_argument('--fgd_n_steps', type=int, default=1)
    parser.add_argument('--oracle_step_size', type=float, default=0.001)
    parser.add_argument('--p', type=float, default=1, help='step size decay exponential')
    parser.add_argument('--oracle_mb_size', type=int, default=128)
    parser.add_argument('--n_global_rounds', type=int, default=100)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--epsilon_adv', type=int, default=8)
    parser.add_argument('--attack_lr', type=float, default=0.005)


    args = parser.parse_args()

    writer = SummaryWriter(
        f'out/{args.dataset}/epsilon{args.epsilon_adv}/{args.weak_learner_hid_dims}/rhog{args.step_size_0}_mb{args.oracle_mb_size}_p{args.p}_{algo}_{ts}'
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    hidden_size = tuple([int(a) for a in args.weak_learner_hid_dims.split("-")])
    epsilon = float(args.epsilon_adv) / 255

    # Load/split training data

    data, label, data_test, label_test, n_class, get_init_weak_learner = load_data(args, hidden_size, device)


    Dx_loss = Dx_losses[args.loss]
    loss = losses[args.loss]
    f_ens = FunctionEnsemble(get_init_function=get_init_weak_learner, device=device)
    delta = None

    for r in tqdm(range(args.n_global_rounds)):
        # attack step
        delta = attack_step(model=f_ens,
                            data=data,
                            label=label,
                            epsilon=epsilon,
                            attack_lr=args.attack_lr
                            )
        # fgd step
        for i in range(args.fgd_n_steps):
            step_size = args.step_size_0/(r*args.fgd_n_steps + i +1)
            fgd_step(f=f_ens,
                     data=(data+delta).detach(),
                     label=label,
                     step_size=step_size,
                     oracle_n_steps=args.oracle_n_steps,
                     oracle_step_size=args.oracle_step_size,
                     oracle_mb_size=args.oracle_mb_size,
                     init_weak_learner=get_init_weak_learner()
                     )
        # test on natural data
        f_data_test = f_ens(data_test)
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
        delta_test = attack_step(model=f_ens,
                            data=data_test,
                            label=label_test,
                            epsilon=epsilon,
                            attack_lr=args.attack_lr
                            )

        f_data_test = f_ens(data_test+delta_test)
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

