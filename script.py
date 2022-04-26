import numpy as np
import os

prefix = './logs/ANN-Same-AFs/'

for file in os.listdir(prefix):
    if '.txt' in file:
        f = open(prefix + file, 'r')
        for line in f:
            splitted = line.split()
            # print(splitted)
            if (splitted and splitted[0] == 'standard' and splitted[2] == 'acc:'):
                standard_max_list = []
                new_line = f.readline().split()
                if (new_line and new_line[0] == 'test'):
                    first_line = new_line[2:]
                    new_string = ''
                    for num in first_line:
                        if '[' in num:
                            new_string += num[1:] + ' '
                        else:
                            new_string += num + ' '
                    for i in range(5):
                        line = f.readline()
                        new_string += line + ' '
                    new_string = new_string[:-5].replace('\n', ' ').replace('   ', ' ').replace('  ', ' ')

                    li = list(new_string.split(' '))
                    if len(li) == 31:
                        li.remove('')
                    max_val = max(li)
                    standard_max_list.append(float(max_val))

            if (splitted and splitted[0] == 'random' and splitted[2] == 'acc:'):
                random_max_list = []
                new_line = f.readline().split()
                if (new_line and new_line[0] == 'test'):
                    first_line = new_line[2:]
                    new_string = ''
                    for num in first_line:
                        if '[' in num:
                            new_string += num[1:] + ' '
                        else:
                            new_string += num + ' '
                    for i in range(5):
                        line = f.readline()
                        new_string += line + ' '
                    new_string = new_string[:-5].replace('\n', ' ').replace('   ', ' ').replace('  ', ' ')

                    li = list(new_string.split(' '))
                    if len(li) == 31:
                        li.remove('')
                    max_val = max(li)
                    random_max_list.append(float(max_val))

            if (splitted and splitted[0] == 'evo' and splitted[2] == 'acc:'):
                evo_max_list = []
                new_line = f.readline().split()
                if (new_line and new_line[0] == 'test'):
                    first_line = new_line[2:]
                    new_string = ''
                    for num in first_line:
                        if '[' in num:
                            new_string += num[1:] + ' '
                        else:
                            new_string += num + ' '
                    for i in range(5):
                        line = f.readline()
                        new_string += line + ' '
                    new_string = new_string[:-5].replace('\n', ' ').replace('   ', ' ').replace('  ', ' ')

                    li = list(new_string.split(' '))
                    if len(li) == 31:
                        li.remove('')
                    max_val = max(li)
                    evo_max_list.append(float(max_val))

            if (splitted and splitted[0] == 'coevo' and splitted[2] == 'acc:'):
                coevo_max_list = []
                new_line = f.readline().split()
                if (new_line and new_line[0] == 'test'):
                    first_line = new_line[2:]
                    new_string = ''
                    for num in first_line:
                        if '[' in num:
                            new_string += num[1:] + ' '
                        else:
                            new_string += num + ' '
                    for i in range(5):
                        line = f.readline()
                        new_string += line + ' '
                    new_string = new_string[:-5].replace('\n', ' ').replace('   ', ' ').replace('  ', ' ')

                    li = list(new_string.split(' '))
                    if len(li) == 31:
                        li.remove('')
                    max_val = max(li)
                    coevo_max_list.append(float(max_val))


        # print(max_list_acc)
        with open(prefix + 'stats.txt', 'a') as text_file:
            # standard_mean = round((sum(standard_max_list) / len(standard_max_list)) * 100, 2)
            # random_mean = round((sum(random_max_list) / len(random_max_list)) * 100, 2)
            evo_mean = round((sum(evo_max_list) / len(evo_max_list)) * 100, 8)
            # coevo_mean = round((sum(coevo_max_list) / len(coevo_max_list)) * 100, 8)
            standard_mean = '0.6'
            random_mean = '0.7'
            coevo_mean = '0.9'
            # evo_mean = '91.3965'
            list_means = {"standard": str(standard_mean), "random": str(random_mean),
                          "evolution": str(evo_mean), "coevo": str(coevo_mean)}
            sorted_list = sorted(list_means.items(), key=lambda elem: elem[1])
            text_file.write(file[:-10] + ':\n\t')
            for i, (key, val) in enumerate(reversed(sorted_list)):
                if i != 3:
                    text_file.write(key + ': ' + val + '\n\t')
                else:
                    text_file.write(key + ': ' + val + '\n\n')
            text_file.close()