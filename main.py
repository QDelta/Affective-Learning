import sklearn.svm as svm
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
    acc_sum = 0.0
    for t in range(DOMAIN_NUM):
        print(f'Training and testing SVM {t + 1}')
        acc = train_test_svm(t)
        print(f'\nAccuracy {acc}\n')
        acc_sum += acc
    avg_acc = acc_sum / DOMAIN_NUM
    print(f'Average accuracy {avg_acc}')
    return avg_acc

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128
LR = 1e-4

def data_transfrom(x):
    return torch.from_numpy(x).float()

def train_dann(dom_for_test, epoches):
    train_data = EEGDataset(dom_for_test, train=True, transform=data_transfrom)
    test_data = EEGDataset(dom_for_test, train=False, transform=data_transfrom)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    model = EEGDANN().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    train_size = len(train_data)
    test_size = len(test_data)
    
    for e in range(epoches):
        print(f'Epoch {e + 1}')

        # Training
        model.train()
        for batch, (x, y, dom) in enumerate(train_loader):
            x, y, dom = x.to(DEVICE), y.to(DEVICE), dom.to(DEVICE)

            y_pred, dom_pred = model(x)
            y_loss = loss_fn(y_pred, y)
            dom_loss = loss_fn(dom_pred, dom)
            loss = y_loss + dom_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss = y_loss.item()
                current = batch * len(x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{train_size:>5d}]")

        # Testing
        num_batches = len(test_loader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for x, y, _ in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)

                pred, _ = model(x)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            test_loss /= num_batches
            correct /= test_size
            print(f'Test Accuracy {(100 * correct):>0.1f}% Avg Loss {test_loss:>8f}\n')

train_dann(0, 1024)