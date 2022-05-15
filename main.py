from os.path import join
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from dann import EEGDANN
from eeg import CLASS_NUM, DOMAIN_NUM, EEGDataset

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 100
LR_INIT = 1e-4
MOMENTUM = 0.3
WEIGHT_DECAY = 1e-2

def data_transfrom(x):
    return torch.from_numpy(x).float()

def train_dann(target_dom, epoches):
    source_data = EEGDataset(target_dom, source=True, transform=data_transfrom)
    target_data = EEGDataset(target_dom, source=False, transform=data_transfrom)
    source_loader = DataLoader(source_data, batch_size=BATCH_SIZE, shuffle=True)
    target_loader = DataLoader(target_data, batch_size=BATCH_SIZE, shuffle=True)

    stat_acc_loss = []

    model = EEGDANN().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(),
        lr=LR_INIT, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    loss_label = nn.CrossEntropyLoss()
    loss_domain = nn.BCELoss()

    source_size = len(source_data)
    target_size = len(target_data)
    source_batchnum = len(source_loader)
    target_batchnum = len(target_loader)

    for e in range(epoches):
        print(f'Data {target_dom} Epoch {e + 1}')

        # Training
        model.train()

        target_data_iter = None
        for batch, (x, y, dom) in enumerate(source_loader):
            # supervised, source domain
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            dom = dom.to(DEVICE)

            # warm start of GRL
            ratio = (batch + e * source_batchnum) / (epoches * source_batchnum)
            ratio = 2.0 / (1.0 + np.exp(-100 * ratio))
            dom_lambda = 2.0 * (1 - ratio)

            y_pred, dom_pred = model(x, label_lambda=1.0, dom_lambda=dom_lambda)
            y_loss = loss_label(y_pred, y)
            dom_loss = loss_domain(dom_pred.flatten(), dom)
            loss = y_loss + dom_loss

            if batch % 100 == 0:
                current = batch * BATCH_SIZE
                print(f"loss: {y_loss.item():>7f} dom_loss: {dom_loss.item():>7f}  [{current+1:>5d}/{source_size:>5d}]")

            # unsupervised, target domain
            if batch % target_batchnum == 0:
                target_data_iter = iter(target_loader)
            x, _, dom = next(target_data_iter)
            x = x.to(DEVICE)
            dom = dom.to(DEVICE)

            _, dom_pred = model(x, label_lambda=0.0, dom_lambda=dom_lambda)
            target_dom_loss = loss_domain(dom_pred.flatten(), dom)

            loss = y_loss + dom_loss + target_dom_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Testing
        model.eval()
        test_loss, correct = 0, 0
        l_correct = np.zeros(CLASS_NUM)
        l_count = np.zeros(CLASS_NUM)
        with torch.no_grad():
            for x, y, _ in target_loader:
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

    return np.array(stat_acc_loss)

if __name__ == '__main__':
    print("[info] Current device:", DEVICE)
    start_time = time.time()
    for d in range(DOMAIN_NUM):
        epoches = 100
        stat_acc_loss = train_dann(d, epoches)
        np.savetxt(join('output', f'acc_loss{d}.txt'), stat_acc_loss)
    end_time = time.time()
    print('[info] Time used:', end_time - start_time)