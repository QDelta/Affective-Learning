from os.path import join
import time
import sklearn.svm as svm
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from dann import EEGDANN
from eeg import DOMAIN_NUM, split_data_for_svm, EEGDataset

def train_test_svm(dom_for_test):
    model = svm.SVC(kernel='linear', verbose=True, shrinking=False)
    train_data, train_label, test_data, test_label = split_data_for_svm(dom_for_test)
    model.fit(train_data, train_label)
    model_output = model.predict(test_data)
    acc_count = (model_output == test_label).sum()
    return acc_count / len(test_label)

def base_line():
    stat_acc = []
    for t in range(DOMAIN_NUM):
        print(f'[info] Training and testing SVM {t + 1}')
        acc = train_test_svm(t)
        print(f'\n[info] Accuracy {acc}\n')
        stat_acc.append(acc)
    stat_acc = np.array(stat_acc)
    print('[info] Accuracy:', stat_acc)
    print('[info] Average accuracy:', np.average(stat_acc))
    return stat_acc

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 100
LR_INIT = 1e-4
MOMENTUM = 0.0
WEIGHT_DECAY = 1e-2

def data_transfrom(x):
    return torch.from_numpy(x).float()

def train_dann(dom_for_test, pre_epoches, epoches):
    train_data = EEGDataset(dom_for_test, train=True, transform=data_transfrom)
    test_data = EEGDataset(dom_for_test, train=False, transform=data_transfrom)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    stat_acc_loss = []

    model = EEGDANN().to(DEVICE)
    pre_optimizer = torch.optim.SGD(model.parameters(),
        lr=LR_INIT, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    optimizer = torch.optim.SGD(model.parameters(),
        lr=LR_INIT, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    train_size = len(train_data)
    test_size = len(test_data)

    for e in range(pre_epoches):
        print(f'Data {dom_for_test} Pre epoch {e + 1}')

        # Training
        model.train()
        for batch, (x, y, dom) in enumerate(train_loader):
            y = y.to(dtype=torch.int64)
            dom = dom.to(dtype=torch.int64)
            x, y, dom = x.to(DEVICE), y.to(DEVICE), dom.to(DEVICE)

            y_pred, dom_pred = model(x, label_lambda=1.0, dom_lambda=1.0)
            y_loss = loss_fn(y_pred, y)
            dom_loss = loss_fn(dom_pred, dom)
            loss = y_loss + dom_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                current = batch * len(x)
                print(f"loss: {y_loss.item():>7f} dom_loss: {dom_loss.item():>7f}  [{current+1:>5d}/{train_size:>5d}]")

        # Testing
        num_batches = len(test_loader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for x, y, _ in test_loader:
                y = y.to(dtype=torch.int64)
                x, y = x.to(DEVICE), y.to(DEVICE)

                pred, _ = model(x)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= test_size
        stat_acc_loss.append((correct, test_loss))
        print(f'Test Accuracy {(100 * correct):>0.1f}% Avg Loss {test_loss:>8f}\n')


    for e in range(epoches):
        print(f'Data {dom_for_test} Epoch {e + 1}')

        # Training
        model.train()
        for batch, (x, y, dom) in enumerate(train_loader):
            y = y.to(dtype=torch.int64)
            dom = dom.to(dtype=torch.int64)
            x, y, dom = x.to(DEVICE), y.to(DEVICE), dom.to(DEVICE)

            # p = float(batch + e * train_batchnum) / (epoches * train_batchnum)
            # lambda_ = -1.0 + 2.0 / (1.0 + np.exp(-10 + p))

            y_pred, dom_pred = model(x, label_lambda=1.0, dom_lambda=-2.0)
            y_loss = loss_fn(y_pred, y)
            dom_loss = loss_fn(dom_pred, dom)
            loss = y_loss + dom_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                current = batch * len(x)
                print(f"loss: {y_loss.item():>7f} dom_loss: {dom_loss.item():>7f}  [{current+1:>5d}/{train_size:>5d}]")

        # Testing
        num_batches = len(test_loader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for x, y, _ in test_loader:
                y = y.to(dtype=torch.int64)
                x, y = x.to(DEVICE), y.to(DEVICE)

                pred, _ = model(x)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= test_size
        stat_acc_loss.append((correct, test_loss))
        print(f'Test Accuracy {(100 * correct):>0.1f}% Avg Loss {test_loss:>8f}\n')

    return np.array(stat_acc_loss)

if __name__ == '__main__':
    print("[info] Current device:", DEVICE)
    start_time = time.time()
    for d in range(DOMAIN_NUM):
        pre_epoches = 0 # not useful
        epoches = 256
        stat_acc_loss = train_dann(d, pre_epoches, epoches)
        np.savetxt(join('output', f'acc_loss{d}.txt'), stat_acc_loss)
    end_time = time.time()
    print('[info] Time used:', end_time - start_time)