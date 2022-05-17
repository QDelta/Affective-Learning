import sklearn.svm as svm
import numpy as np
from eeg import DOMAIN_NUM, split_data

def train_test_sklearn(target_dom, skl_model):
    train_data, train_label, test_data, test_label = split_data(target_dom)
    skl_model.fit(train_data, train_label)
    model_output = skl_model.predict(test_data)
    acc_count = (model_output == test_label).sum()
    return acc_count / len(test_label)

def base_line(model_type):
    stat_acc = []
    for t in range(DOMAIN_NUM):
        if (model_type == 'SVM'):
            model = svm.SVC(kernel='linear', verbose=True, shrinking=False)
        else:
            print(f'[Error] Unsupported model type {model_type}, expect SVM')
        print(f'[info] Training and testing {model_type} {t + 1}')
        acc = train_test_sklearn(t, model)
        print(f'\n[info] Accuracy {acc}\n')
        stat_acc.append(acc)
    stat_acc = np.array(stat_acc)
    print('[info] Accuracy:', stat_acc)
    print('[info] Average accuracy:', np.average(stat_acc))
    return stat_acc

if __name__ == '__main__':
    base_line('SVM')