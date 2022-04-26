from utils import random_string, write_params, fprint, conv_net, vanilla_net, print_losses_acc_to_file
from utils import get_args, get_dataset_name, get_af, write_summary
from cartesian.algorithm import oneplus
from data import load_dataset
from cartesian.cgp import *
from primitives import pset
from os.path import exists
from os import makedirs
import torch.nn as nn
import numpy as np
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
best_coevo_l1_af = None
best_coevo_l2_af = None
best_coevo_l3_af = None
best_random_afs = None
best_evo_single_afs = None
best_evo_triple_afs = None
best_random_score = float('-inf')
best_evo_single_score = float('-inf')
best_evo_triple_score = float('-inf')
best_coevo_score = float('-inf')
mode_flag = None
res_dir = None
n_dataset = None
n_replicates = None
n_baseline_af = None
baseline_af = None
output_dim = 10
n_iter = None
n_epochs = None
n_mini_epochs = 3
batch_size = None
all_train_loader = None
train_loader = None
val_loader = None
test_loader = None
file_name = None
cnn = None

def initialize_params():
    global best_coevo_l1_af, best_coevo_l2_af, best_coevo_l3_af, best_coevo_score
    global best_random_afs, best_random_score
    global best_evo_single_afs, best_evo_single_score
    global best_evo_triple_afs, best_evo_triple_score
    global mode_flag
    _, af = get_af(n_baseline_af)
    best_coevo_l1_af = af
    best_coevo_l2_af = af
    best_coevo_l3_af = af
    best_random_afs = [af, af, af]
    best_evo_single_afs = [af, af, af]
    best_evo_triple_afs = [af, af, af]
    best_random_score = float('-inf')
    best_evo_single_score = float('-inf')
    best_evo_triple_score = float('-inf')
    best_coevo_score = float('-inf')
    mode_flag = -1

def network_score(net):
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = 0.01)

    epochs_train_acc = np.zeros(n_epochs)
    epochs_train_losses = np.zeros(n_epochs)
    epochs_test_acc = np.zeros(n_epochs)
    epochs_test_losses = np.zeros(n_epochs)
    for epoch in range(n_epochs):
        train_batch_loss = 0
        test_batch_loss = 0
        test_true = 0
        test_total = 0
        net.train()
        for i, (images, labels) in enumerate(all_train_loader):
            images = images.to(device)
            optimizer.zero_grad()
            y_pred = net(images)
            train_loss = criterion(y_pred, labels.to(device))

            predictions = torch.argmax(y_pred, dim=1)
            test_total += labels.shape[0]
            test_true += torch.sum((predictions == labels.to(device)).int())

            train_batch_loss += train_loss.item()
            train_loss.backward()
            optimizer.step()

        epochs_train_losses[epoch] = train_batch_loss / len(all_train_loader)
        epochs_train_acc[epoch] = test_true / test_total
        print(f'Training loss for epoch #{epoch}: {epochs_train_losses[epoch]:.4f}')
        print(f'Training accuracy for epoch #{epoch}: {epochs_train_acc[epoch]:.4f}')

        test_total = 0
        test_true = 0

        net.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                y_pred = net(images)

                predictions = torch.argmax(y_pred, dim = 1)
                test_total += labels.shape[0]
                test_true += torch.sum((predictions == labels.to(device)).int())

                test_loss = criterion(y_pred, labels.to(device))
                test_batch_loss += test_loss.item()

            epochs_test_losses[epoch] = test_batch_loss / len(test_loader)
            epochs_test_acc[epoch] = test_true / test_total
            print(f'Test loss for epoch #{epoch}: {epochs_test_losses[epoch]:.4f}')
            print(f'Test accuracy for epoch #{epoch}: {epochs_test_acc[epoch]:.4f}')

    print_losses_acc_to_file(file_name, epochs_train_losses, epochs_test_losses, epochs_train_acc, epochs_test_acc, mode_flag)

    return max(epochs_test_acc)

def standard_fc():
    _, af = get_af(n_baseline_af)
    activation_functions = [af, af, af]
    if cnn:
        net = conv_net(activation_functions, output_dim)
    else:
        net = vanilla_net(activation_functions, output_dim)
    score = network_score(net)
    return score

def random_fc():
    global best_random_afs, mode_flag
    mode_flag = 0
    af_cartesian = Cartesian('Random', pset, n_rows=5, n_columns=5, n_back=5)
    oneplus(random_fitness, f_tol=0.01, cls=af_cartesian, maxiter = 2, lambda_= n_iter * 3)
    activation_functions = best_random_afs
    if cnn:
        best_random_net = conv_net(activation_functions, output_dim)
    else:
        best_random_net = vanilla_net(activation_functions, output_dim)
    score = network_score(best_random_net)
    return score

def evo_single_fc():
    global best_evo_single_afs, mode_flag
    mode_flag = 1
    af_cartesian = Cartesian('Evo-Single', pset, n_rows=5, n_columns=5, n_back=5)
    oneplus(evo_single_fitness, f_tol=0.01, cls=af_cartesian, maxiter=n_iter * 3)
    activation_functions = best_evo_single_afs
    if cnn:
        best_evo_single_net = conv_net(activation_functions, output_dim)
    else:
        best_evo_single_net = vanilla_net(activation_functions, output_dim)
    score = network_score(best_evo_single_net)
    return score

def evo_triple_fc():
    global best_evo_triple_afs, mode_flag
    mode_flag = 2
    af_cartesian = Cartesian('Evo-Triple', pset, n_rows=5, n_columns=5, n_back=5)
    oneplus(evo_triple_fitness, f_tol=0.01, cls=af_cartesian, maxiter=n_iter * 3)
    activation_functions = best_evo_triple_afs
    if cnn:
        best_evo_net = conv_net(activation_functions, output_dim)
    else:
        best_evo_net = vanilla_net(activation_functions, output_dim)
    score = network_score(best_evo_net)
    return score

def coevo_fc():
    global mode_flag
    mode_flag = 3
    af_cartesian = Cartesian('Coevo', pset, n_rows=5, n_columns=5, n_back=5)
    oneplus(coevo_fitness, f_tol=0.01, cls=af_cartesian, maxiter=n_iter)
    mode_flag = 4
    oneplus(coevo_fitness, f_tol=0.01, cls=af_cartesian, maxiter=n_iter)
    mode_flag = 5
    oneplus(coevo_fitness, f_tol=0.01, cls=af_cartesian, maxiter=n_iter)
    if cnn:
        best_coevo_net = conv_net([best_coevo_l1_af, best_coevo_l2_af, best_coevo_l3_af], output_dim)
    else:
        best_coevo_net = vanilla_net([best_coevo_l1_af, best_coevo_l2_af, best_coevo_l3_af], output_dim)
    score = network_score(best_coevo_net)
    return score


def random_fitness(individual):
    global best_random_afs
    print(f'AF: {to_polish(individual)[0][0]}')
    func = compile(individual)
    activation_functions = best_random_afs
    rand = np.random.randint(0, 3)
    activation_functions = activation_functions[:rand] + [func] + activation_functions[rand + 1:]
    if cnn:
        net = conv_net(activation_functions, output_dim)
    else:
        net = vanilla_net(activation_functions, output_dim)
    result, is_best = fitness(net)
    if is_best:
        best_random_afs = activation_functions
    return result

def evo_single_fitness(individual):
    global best_evo_single_afs
    print(f'AF: {to_polish(individual)[0][0]}')
    func = compile(individual)
    activation_functions = [func, func, func]
    if cnn:
        net = conv_net(activation_functions, output_dim)
    else:
        net = vanilla_net(activation_functions, output_dim)
    result, is_best = fitness(net)
    if is_best:
        best_evo_single_afs = activation_functions
    return result

def evo_triple_fitness(individual):
    global best_evo_triple_afs
    print(f'AF: {to_polish(individual)[0][0]}')
    func = compile(individual)
    activation_functions = best_evo_triple_afs
    rand = np.random.randint(0, 3)
    activation_functions = activation_functions[:rand] + [func] + activation_functions[rand+1:]
    if cnn:
        net = conv_net(activation_functions, output_dim)
    else:
        net = vanilla_net(activation_functions, output_dim)
    result, is_best = fitness(net)
    if is_best:
        best_evo_triple_afs = activation_functions
    return result

def coevo_fitness(individual):
    global best_coevo_l1_af, best_coevo_l2_af, best_coevo_l3_af
    func = compile(individual)
    if mode_flag == 3:
        activation_functions = [func, best_coevo_l2_af, best_coevo_l3_af]
    elif mode_flag == 4:
        activation_functions = [best_coevo_l1_af, func, best_coevo_l3_af]
    elif mode_flag == 5:
        activation_functions = [best_coevo_l1_af, best_coevo_l2_af, func]
    if cnn:
        net = conv_net(activation_functions, output_dim)
    else:
        net = vanilla_net(activation_functions, output_dim)
    result, is_best = fitness(net)
    if is_best and mode_flag == 3:
        print(f'Best Input AF: {to_polish(individual)[0][0]}')
        best_coevo_l1_af = func
    elif is_best and mode_flag == 4:
        print(f'Best Hidden AF: {to_polish(individual)[0][0]}')
        best_coevo_l2_af = func
    elif is_best and mode_flag == 5:
        print(f'Best Output AF: {to_polish(individual)[0][0]}')
        best_coevo_l3_af = func
    return result

def fitness(net):
    global  best_random_score, best_evo_single_score, best_evo_triple_score, best_coevo_score
    net.to(device)
    flag_best = False
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    epochs_train_acc = np.zeros(n_mini_epochs)
    epochs_train_losses = np.zeros(n_mini_epochs)
    epochs_val_acc = np.zeros(n_mini_epochs)
    epochs_val_losses = np.zeros(n_mini_epochs)

    for epoch in range(n_mini_epochs):
        train_batch_loss = 0
        val_batch_loss = 0
        test_true = 0
        test_total = 0
        net.train()
        for images, labels in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            y_pred = net(images)

            predictions = torch.argmax(y_pred, dim=1)
            test_total += labels.shape[0]
            test_true += torch.sum((predictions == labels.to(device)).int())

            train_loss = criterion(y_pred, labels.to(device))

            train_batch_loss += train_loss.item()

            train_loss.backward()
            optimizer.step()

        epochs_train_losses[epoch] = train_batch_loss / len(train_loader)
        epochs_train_acc[epoch] = test_true / test_total
        print(f'Training loss for epoch #{epoch}: {epochs_train_losses[epoch]:.4f}')
        print(f'Training accuracy for epoch #{epoch}: {epochs_train_acc[epoch]:.4f}')

        test_total = 0
        test_true = 0

        net.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                y_pred = net(images)

                predictions = torch.argmax(y_pred, dim=1)
                test_total += labels.shape[0]
                test_true += torch.sum((predictions == labels.to(device)).int()).item()

                val_score = criterion(y_pred, labels.to(device))

                val_batch_loss += val_score.item()

            epochs_val_losses[epoch] = val_batch_loss / len(val_loader)
            epochs_val_acc[epoch] = test_true / test_total
            print(f'Test loss for epoch #{epoch}: {epochs_val_losses[epoch]:.4f}')
            print(f'Test accuracy for epoch #{epoch}: {epochs_val_acc[epoch]:.4f}')

        if mode_flag == 0 and epochs_val_acc[epoch] > best_random_score:
            best_random_score = epochs_val_acc[epoch]
            flag_best = True

        elif mode_flag == 1 and epochs_val_acc[epoch] > best_evo_single_score:
            best_evo_single_score = epochs_val_acc[epoch]
            flag_best = True

        ###
        elif mode_flag == 2 and epochs_val_acc[epoch] > best_evo_triple_score:
            best_evo_triple_score = epochs_val_acc[epoch]
            flag_best = True

        elif mode_flag >= 3 and epochs_val_acc[epoch] > best_coevo_score:
            best_coevo_score = epochs_val_acc[epoch]
            flag_best = True
        else:
            pass

    return max(epochs_val_acc), flag_best

def main():
    global res_dir, n_dataset, n_replicates, n_baseline_af, n_iter, n_epochs, output_dim, cnn
    global all_train_loader, train_loader, val_loader, test_loader, batch_size, file_name
    res_dir, n_dataset, n_replicates, n_epochs, n_baseline_af, n_iter, train1_size, train2_size, cnn = get_args()
    dataset_name = get_dataset_name(n_dataset)
    baseline_af_name, baseline_af = get_af(n_baseline_af)

    if not exists(res_dir):
        makedirs(res_dir)

    file_name = res_dir + dataset_name + '_' + baseline_af_name + '_' + random_string() + '.txt'

    write_params(file_name, device, dataset_name, n_replicates, n_epochs, n_iter, baseline_af_name, train1_size, train2_size, cnn)

    scores = {'standard': [], 'random': [], 'evo-single': [], 'evo-triple': [], 'coevo': []}

    for rep in range(n_replicates):
        initialize_params()
        all_train_loader, train_loader, val_loader, test_loader, batch_size = load_dataset(n_dataset, train1_size, train2_size)

        score_standard_fc = standard_fc()
        scores['standard'].append(score_standard_fc)
        fprint(file_name, f'\n Replicate {rep}, standard fc:\n\t score - {score_standard_fc}')

        score_random_fc = random_fc()
        scores['random'].append(score_random_fc)
        fprint(file_name, f'\n Replicate {rep}, random fc:\n\t score - {score_random_fc}')

        score_evo_single_fc = evo_single_fc()
        scores['evo-single'].append(score_evo_single_fc)
        fprint(file_name, f'\n Replicate {rep}, evo fc:\n\t score - {score_evo_single_fc}')

        score_evo_triple_fc = evo_triple_fc()
        scores['evo-triple'].append(score_evo_triple_fc)
        fprint(file_name, f'\n Replicate {rep}, evo fc:\n\t score - {score_evo_triple_fc}')

        score_coevo_fc = coevo_fc()
        scores['coevo'].append(score_coevo_fc)
        fprint(file_name, f'\n Replicate {rep}, coevo fc:\n\t score - {score_coevo_fc}')

    print(f'############################################################')
    print(f'Finished! the results are written in `{res_dir}` directory.')
    print(f'############################################################')

    fprint(file_name, f'\n\n Summary of {n_replicates} replicates:\n\t')

    write_summary(scores, file_name)

if __name__ == '__main__':
    main()
