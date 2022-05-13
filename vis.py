import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from eeg import DOMAIN_NUM

if __name__ == '__main__':
    for d in range(DOMAIN_NUM):
        stat_acc_loss = np.loadtxt(join('output', f'acc_loss{d}.txt'))
        epoch = []
        acc = []
        loss = []
        for i, v in enumerate(stat_acc_loss):
            epoch.append(i + 1)
            acc.append(v[0])
            loss.append(v[1])
        plt.plot(epoch, acc)
        plt.plot(epoch, loss)
        plt.title(f'test: {d}')
        plt.show()