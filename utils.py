from neural_network import ConvNet, NeuralNet
from mlxtend.evaluate import permutation_test
from argparse import ArgumentParser
from argparse import BooleanOptionalAction
from string import ascii_lowercase
from statistics import median, mean
from sys import stdin, exit
from random import choices
from pathlib import Path
import torch.nn as nn
import numpy as np

def get_args():
    parser = ArgumentParser(description='Coevolution Activation Functions')
    parser.add_argument('--dir', dest='res_dir', type=str, action='store', help='directory where results are placed')
    parser.add_argument('--ds', dest='n_dataset', type=int, action='store', help='dataset num: 1. MNIST, 2. FashionMNIST, 3. KMNIST, 4. USPS')
    parser.add_argument('--rep', dest='n_replicates', type=int, action='store', help='number of replicate runs')
    parser.add_argument('--n_epochs', dest='n_epochs', type=int, action='store', help='number of epochs')
    parser.add_argument('--n_baseline_af', dest='n_baseline_af', type=int, action='store', help='AF baseline: 1. ReLU, 2. LeakyReLU')
    parser.add_argument('--n_iter', dest='n_iter', type=int, action='store', help='number of generations (evolution)')
    parser.add_argument('--n_train1', dest='n_train1', type=float, action='store', help='Train1 set size as a fraction')
    parser.add_argument('--n_train2', dest='n_train2', type=float, action='store', help='Train2 set size as a fraction')
    parser.add_argument('--cnn', dest='cnn', action=BooleanOptionalAction)
    args = parser.parse_args()
    if None in [getattr(args, arg) for arg in vars(args)]:
        parser.print_help()
        exit()
    res_dir, n_dataset, n_replicates, n_epochs, n_baseline_af, n_iter, n_train1, n_train2, cnn = args.res_dir + '/', \
                                                                                                 args.n_dataset, \
                                                                                                 args.n_replicates,\
                                                                                                 args.n_epochs,\
                                                                                                 args.n_baseline_af,\
                                                                                                 args.n_iter,\
                                                                                                 args.n_train1,\
                                                                                                 args.n_train2,\
                                                                                                 args.cnn
    return res_dir, n_dataset, n_replicates, n_epochs, n_baseline_af, n_iter, n_train1, n_train2, cnn

def get_dataset_name(n_dataset):
    if n_dataset == 1:
        return 'MNIST'
    elif n_dataset == 2:
        return 'FashionMNIST'
    elif n_dataset == 3:
        return 'KMNIST'
    elif n_dataset == 4:
        return 'USPS'
    elif n_dataset == 5:
        return 'ArabicHWD'
    else:
        raise Exception("No such dataset!")

def get_af(n_baseline_af):
    if n_baseline_af == 1:
        return 'ReLU', nn.ReLU()
    elif n_baseline_af == 2:
        return 'LeakyReLU', nn.LeakyReLU()
    else:
        raise Exception("No such af!")

def random_string():
    return ''.join(choices(ascii_lowercase, k=5))

def fprint(fname, s):
    if stdin.isatty(): # running interactively
        print(s)
    with open(Path(fname), 'a') as f:
        f.write(s)

def write_params(file_name, device, ds_name, n_replicates, n_epochs, n_iter, baseline_af_name, train1_size, train2_size, cnn):
    fprint(file_name, f' Device: {device}\n'
                      f' Dataset name: {ds_name}\n'
                      f' Train1 size (fraction): {train1_size}\n'
                      f' Train2 size (fraction): {train2_size}\n'
                      f' Architecture: {"CNN" if cnn else "FCN"}\n'
                      f' Num of replicates: {n_replicates}\n'
                      f' Number of epochs: {n_epochs}\n'
                      f' Baseline activation function: {baseline_af_name}\n'
                      f' Number of generations {n_iter}\n')

def conv_net(activation_functions, output_dim):
    net = ConvNet(activation_functions, output_dim)
    return net

def vanilla_net(activation_functions, output_dim):
    net = NeuralNet(activation_functions, output_dim)
    return net

def write_summary(scores, file_name):
    means = []
    for (net, score) in scores.items():
        if len(score) == 0:
            break
        elif len(score) == 1:
            means.append((net, score, score))
        else:
            means.append((net, mean(score), score))
    means = sorted(means, key=lambda x: x[1], reverse=True)
    s = ''
    for (net, score, _) in means:
        s += net + ': ' + str(score) + '\n\t'
    fprint(file_name, s + '\n')

def permutation_tests(scores, file_name):
    n_rounds = 10000
    th1, th2 = 0.0001, 0.05
    medians = []
    for (net, score) in scores.items():
        medians.append((net, median(score), score))
    n_scores = len(medians)

    # Medians is a list of tuples ->
    # [('standard', median, [rep1, rep2]), ('random', median...)]

    print(f'Medians: {medians}')
    med = round(medians[2][1], 3)
    s = f'Coevo: {med} vs. '
    for i in range (n_scores - 1):
        net = medians[i][0]                 # name of net
        med_vs = round(medians[i][1], 3)    # median of vs net
        pval = permutation_test(medians[i][2], medians[2][2], method='approximate',
                                num_rounds=n_rounds, func=lambda x, y: np.abs(np.median(x) - np.median(y)))
        pv = '!!' if pval < th1 else '!' if pval < th2 else '\b'
        pval = round(pval, 4)
        s += f'{net}: {med_vs} (p {pval} {pv}), '
    fprint(file_name, s[:-2] + '\n')

    # rank networks, do permutation testing of rank i vs lower rank i + 1
    medians = sorted(medians, key=lambda x: x[1], reverse=True)
    s = ' sorted, '
    for i in range(n_scores):
        net = medians[i][0]
        med = round(medians[i][1], 3)
        if i < n_scores - 1:
            pval = permutation_test(medians[i][2], medians[i+1][2], method='approximate', num_rounds=n_rounds, \
                                    func=lambda x, y: np.abs(np.median(x) - np.median(y)))
            pv = '!!' if pval < th1 else '!' if pval < th2 else '\b'
            pval = round(pval, 4)
            s += f'{net}: {med} (p {pval} {pv}), '
        else:
            s += f'{net}: {med}'
    fprint(file_name, s + '\n')

def print_losses_acc_to_file(file_name, epochs_train_losses, epochs_test_losses, epochs_train_acc, epochs_test_acc, mode_flag):
    if mode_flag == -1:
        net = 'standard'
    elif mode_flag == 0:
        net = 'random'
    elif mode_flag == 1:
        net = 'evo'
    elif mode_flag >= 2:
        net = 'coevo'

    fprint(file_name, f'\n\n\t {net} net, losses:\n\t train - {epochs_train_losses}')
    fprint(file_name, f'\n\n\t {net} net, losses:\n\t test - {epochs_test_losses}')
    fprint(file_name, f'\n\n\t {net} net, acc:\n\t train - {epochs_train_acc}')
    fprint(file_name, f'\n\n\t {net} net, acc:\n\t test - {epochs_test_acc}\n')