import numpy as np
import sklearn.svm as svm
import torch
from torch import nn
from dann import EEGDANN
from eeg import DOMAIN_NUM, split_data, split_data_for_svm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_test_svm(dom_for_test):
    model = svm.SVC(kernel='linear', verbose=True, shrinking=False)
    train_data, train_label, test_data, test_label = split_data_for_svm(dom_for_test)
    model.fit(train_data, train_label)
    model_output = model.predict(test_data)
    acc_count = (model_output == test_label).sum()
    return acc_count / len(test_label)

def base_line():
    acc_sum = 0.0
    for t in range(DOMAIN_NUM):
        print(f'Training and testing SVM {t + 1}')
        acc = train_test_svm(t)
        print(f'\nAccuracy {acc}\n')
        acc_sum += acc
    avg_acc = acc_sum / DOMAIN_NUM
    print(f'Average accuracy {avg_acc}')
    return avg_acc

def train_dann(dom_for_test, epoches):
    model = EEGDANN().to(DEVICE)
    train_data, test_data = split_data(dom_for_test)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-4)
    
    for e in range(epoches):
        print(f'Epoch {e + 1}')
        for x, (y, dom) in train_data:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            dom = dom.to(DEVICE)

            y_pred, dom_pred = model(x)
            y_loss = loss_fn(y_pred, y)
            dom_loss = loss_fn(dom_pred, dom)
            loss = y_loss + dom_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_loss, acc_count = 0, 0
        with torch.no_grad():
            for x, y in test_data:
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                y_pred, _ = model(x)
                test_loss += loss_fn(y_pred, y).item()
                acc_count += (y_pred.argmax() == y).type(torch.float).sum().item()
            avg_loss = test_loss / len(test_data)
            acc = acc_count / len(test_data)
            print(f'Test Accuracy {acc} Avg Loss {avg_loss}')

train_dann(0, 10)