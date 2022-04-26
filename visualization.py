import matplotlib.pyplot as plt

# coevo_test_acc = [0.53000003, 0.80275005, 0.88400006, 0.91150004,
#                   0.92025006, 0.93700004, 0.94100004, 0.94450003,
#                   0.94850004, 0.94225007, 0.94825006, 0.94350004,
#                   0.94875002, 0.95175004, 0.94925004, 0.95500004,
#                   0.95025003, 0.94825006, 0.94900006, 0.93825006,
#                   0.94450003, 0.94950002, 0.95350003, 0.95425004,
#                   0.95525002, 0.95050007, 0.95375007, 0.95375007,
#                   0.95150006, 0.95775002]
#
# coevo_train_acc = [0.33329165, 0.69637501, 0.84679168, 0.90458333,
#                    0.9273333,  0.94008332, 0.9515, 0.95766664,
#                    0.96099997, 0.96454167, 0.96820831, 0.96883333,
#                    0.97112501, 0.9740833,  0.97675002, 0.97858334,
#                    0.98266667, 0.98554164, 0.98491663, 0.97179168,
#                    0.97095835, 0.9795,     0.98316664, 0.98724997,
#                    0.98837501, 0.98787498, 0.98791665, 0.98745835,
#                    0.99066663, 0.99295831]
#
# coevo_train_loss = [1.85376838, 0.86324211, 0.49579956, 0.32661402,
#                     0.2520451,  0.20590947, 0.16895189, 0.14218104,
#                     0.12710221, 0.1188393,  0.10420191, 0.10092854,
#                     0.09379219, 0.08431609, 0.07644634, 0.0701442,
#                     0.05672987, 0.05104314, 0.04768905, 0.09538717,
#                     0.09642693, 0.06640585, 0.05509304, 0.0440667,
#                     0.03938497, 0.03902362, 0.03868801, 0.03970656,
#                     0.03005086, 0.02350752]
#
# coevo_test_loss = [1.19552004, 0.62188631, 0.38701707, 0.30527818,
#                    0.27380382, 0.23153647, 0.20954964, 0.19573142,
#                    0.1825624,  0.21455853, 0.20305145, 0.21327635,
#                    0.20974851, 0.18565702, 0.20308971, 0.18732087,
#                    0.23650195, 0.20804503, 0.22830549, 0.26800951,
#                    0.23210163, 0.2246686,  0.21324389, 0.22474241,
#                    0.23347427, 0.24894883, 0.23781312, 0.25879826,
#                    0.25737288, 0.26440043]
#
# standard_train_loss = [2.08564029, 1.3765379,  0.91580612, 0.64160751,
#                        0.46147189, 0.37118212, 0.30245906, 0.25939355,
#                        0.22756048, 0.20426658, 0.18767732, 0.16808037,
#                        0.15261557, 0.13906426, 0.1389462,  0.13110886,
#                        0.11813609, 0.1036556, 0.10139197, 0.09837399,
#                        0.09609483, 0.09689444, 0.08975334, 0.08090921,
#                        0.07687422, 0.06402891, 0.06004159, 0.06079976,
#                        0.06057934, 0.05260511]
#
# standard_test_loss = [1.66567451, 1.08603525, 0.83629596, 0.52107799,
#                       0.44556238, 0.3726311, 0.31579092, 0.28835724,
#                       0.2792664,  0.26637994, 0.25240495, 0.24411535,
#                       0.23436337, 0.24079718, 0.26564595, 0.22939398,
#                       0.22686373, 0.23303501, 0.22740207, 0.28794616,
#                       0.25031944, 0.23638539, 0.25096399, 0.25126878,
#                       0.24573348, 0.24856453, 0.25237347, 0.26967427,
#                       0.24969527, 0.25972493]
#
# standard_train_acc = [0.21841666, 0.46291667, 0.67124999, 0.78945833,
#                       0.86708331, 0.89475, 0.912875,   0.92537498,
#                       0.93529165, 0.94199997, 0.94679165, 0.95220834,
#                       0.95700002, 0.95999998, 0.95995831, 0.96179163,
#                       0.96533334, 0.97004163, 0.96999997, 0.97104168,
#                       0.97137499, 0.97187501, 0.97262502, 0.97616667,
#                       0.97795832, 0.98145831, 0.98287499, 0.98212498,
#                       0.98229164, 0.98491663]
#
# standard_test_acc = [0.38750002, 0.61450005, 0.69500005, 0.85825002,
#                       0.88000005, 0.89625007, 0.91300005, 0.91850007,
#                       0.92200005, 0.92775005, 0.92750007, 0.93150002,
#                       0.93650007, 0.93675005, 0.93125004, 0.93750006,
#                       0.94025004, 0.94150007, 0.94050002, 0.93225002,
#                       0.93650007, 0.94350004, 0.94325006, 0.94225007,
#                       0.94475007, 0.94475007, 0.94550002, 0.94375002,
#                       0.94650006, 0.94675004]
#
#
# plt.plot(coevo_train_loss, 'b--', label='coevo train')
# plt.plot(coevo_test_loss, 'g--', label='coevo test')
# plt.plot(standard_train_loss, 'r-', label='standard train')
# plt.plot(standard_test_loss, 'c-', label='standard test')
# plt.title('Coevolution losses')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig('MNIST_loss')
#
# plt.close()
#
# plt.plot(coevo_train_acc, 'b--', label='coevo train')
# plt.plot(coevo_test_acc, 'g--', label='coevo test')
# plt.plot(standard_train_acc, 'r-', label='standard train')
# plt.plot(standard_test_acc, 'c-', label='standard test')
# plt.title('Coevolution accuracies')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.savefig('MNIST_acc')

import numpy as np

x = np.linspace(-10, 10, 100)
y = np.maximum(x, 0)

plt.plot(x, y)
plt.title('ReLU', fontsize=20)
plt.xlabel('x', fontsize=15)
plt.xticks([-10, 0, 10], fontsize=15)
plt.yticks([0, 5, 10], fontsize=15)
plt.ylabel('y', fontsize=15)
plt.savefig('Relu.png')

plt.close()

y = 1 / (1 + np.exp(-x) )

plt.plot(x, y)
plt.title('Sigmoid', fontsize=20)
plt.xlabel('x', fontsize=15)
plt.xticks([-10, 0, 10], fontsize=15)
plt.yticks([0, 0.5, 1], fontsize=15)
plt.ylabel('y', fontsize=15)
plt.savefig('Sigmoid.png')

plt.close()

y = ( 2 / (1 + np.exp(-2*x) ) ) -1

plt.plot(x, y)
plt.title('Tanh', fontsize=20)
plt.xlabel('x', fontsize=15)
plt.xticks([-10, 0, 10], fontsize=15)
plt.yticks([-1, 0, 1], fontsize=15)
plt.ylabel('y', fontsize=15)
plt.savefig('Tanh.png')

plt.close()

y = np.maximum(0.1 * x, x)

plt.plot(x, y)
plt.title('LeakyReLU', fontsize=20)
plt.xlabel('x', fontsize=15)
plt.xticks([-10, 0, 10], fontsize=15)
plt.yticks([-1, 5, 10], fontsize=15)
plt.ylabel('y', fontsize=15)
plt.savefig('LeakyReLU.png')

plt.close()