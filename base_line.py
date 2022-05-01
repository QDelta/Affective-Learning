import sklearn.svm as svm
import numpy as np
from eeg import DOMAIN_NUM, split_data_for_svm

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

if __name__ == '__main__':
    base_line()