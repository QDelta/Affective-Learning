from os.path import join
import time
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from dgdann import EEGDGDANN
from eeg import CLASS_NUM, DOMAIN_NUM, EEGDatasetDG

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 100
LR_INIT = 1e-4
MOMENTUM = 0.0
WEIGHT_DECAY = 1e-2

def data_transfrom(x):
    return torch.from_numpy(x).float()

def train_dg_dann(test_dom: int, epoches: int):
    train_data = EEGDatasetDG(test_dom, train=True, transform=data_transfrom)
    test_data = EEGDatasetDG(test_dom, train=False, transform=data_transfrom)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    stat_acc_loss = []

    model = EEGDGDANN().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(),
        lr=LR_INIT, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    loss_label = nn.CrossEntropyLoss()
    loss_domain = nn.CrossEntropyLoss()

    source_size = len(train_data)
    target_size = len(test_data)
    source_batchnum = len(train_loader)
    target_batchnum = len(test_loader)

    best_acc = 0.0
    best_model = None

    for e in range(epoches):
        print(f'Target {test_dom} Epoch {e + 1}')

        # Training
        model.train()

        for batch, (x, y, dom) in enumerate(train_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            dom = dom.to(DEVICE)

            # warm start
            ratio = (batch + e * source_batchnum) / (epoches * source_batchnum)
            ratio = 2.0 / (1.0 + np.exp(-100 * ratio))
            dom_lambda = 2.0 * (1 - ratio)

            y_pred, dom_pred = model(x, label_lambda=1.0, dom_lambda=dom_lambda)
            y_loss = loss_label(y_pred, y)
            dom_loss = loss_domain(dom_pred, dom)

            if batch % 100 == 0:
                current = batch * BATCH_SIZE
                print(f"loss: {y_loss.item():>7f} dom_loss: {dom_loss.item():>7f}  [{current+1:>5d}/{source_size:>5d}]")

            loss = y_loss + dom_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Testing
        model.eval()
        test_loss, correct = 0, 0
        l_correct = np.zeros(CLASS_NUM)
        l_count = np.zeros(CLASS_NUM)
        with torch.no_grad():
            for x, y, _ in test_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                pred, _ = model(x)
                test_loss += loss_label(pred, y).item()
                pred_label = pred.argmax(1)
                correct += (pred_label == y).type(torch.float).sum().item()

                for i in range(CLASS_NUM):
                    l_correct[i] += (pred_label == y)[y == i].type(torch.float).sum().item()
                    l_count[i] += (y == i).type(torch.float).sum().item()

        print(l_correct / l_count)

        test_loss /= target_batchnum
        correct /= target_size
        stat_acc_loss.append((correct, test_loss))
        print(f'Test Accuracy {(100 * correct):>0.1f}% Avg Loss {test_loss:>8f}\n')

        if correct > best_acc:
            best_acc = correct
            best_model = deepcopy(model.state_dict())

    return np.array(stat_acc_loss), best_model, best_acc

if __name__ == '__main__':
    print("[info] Current device:", DEVICE)
    start_time = time.time()
    accs = []
    for d in range(DOMAIN_NUM):
        epoches = 200
        stat_acc_loss, model, acc = train_dg_dann(d, epoches)
        np.savetxt(join('output', f'acc_loss{d}.txt'), stat_acc_loss)
        accs.append(acc)
    accs = np.array(accs)
    print('[info] Accuracy:', accs)
    print('[info] Average acc:', np.average(accs))
    print('[info] Acc std:', np.std(accs))
    end_time = time.time()
    print('[info] Time used:', end_time - start_time)