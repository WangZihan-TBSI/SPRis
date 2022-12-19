# Course project source code
# Copyright @ Siqi Ping (2022214047) Zihan Wang (2022312876)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer
import random

def pre_processing(X_train, X_test, logTrnasform):
    pt = PowerTransformer()
    ss = StandardScaler()
    if logTrnasform == 1:
        # non-linear transform
        X_train = pd.DataFrame(pt.fit_transform(X_train))
        X_test = pd.DataFrame(pt.transform(X_test))
    else:
        # Standard scalar
        X_train = pd.DataFrame(ss.fit_transform(X_train))
        X_test = pd.DataFrame(ss.transform(X_test))
    return X_train, X_test


def import_original_data():
    train_data = pd.read_csv(
        '/Users/zihan/Seafile/Learning/研究生课程/ML2022/Project/Stroke_prediction_system_Data/Prf_feature_train.csv')
    test_data = pd.read_csv(
        '/Users/zihan/Seafile/Learning/研究生课程/ML2022/Project/Stroke_prediction_system_Data/Prf_feature_test.csv')
    train_label = pd.read_csv(
        '/Users/zihan/Seafile/Learning/研究生课程/ML2022/Project/Stroke_prediction_system_Data/Stroke_label_train.csv')
    # Select relavent features
    selected_features = pd.read_csv('./Feature_selected.csv')
    # 筛选所使用的数据
    X_train = train_data[selected_features['Feature']]
    X_test = test_data[selected_features['Feature']]
    # 只保留有完整记录数据
    y_train = train_label[X_train['DISPCODE'] == 1100]
    X_train = X_train[X_train['DISPCODE'] == 1100]
    X_test = X_test[X_test['DISPCODE'] == 1100]
    # 删除数据记录完整度的栏目
    del X_test['DISPCODE']
    del X_train['DISPCODE']
    print("Data Loading Complete")
    # balance the training data
    number_of_stroke_sample = np.sum(y_train['Stroke'] == 1)
    X_train_stroke = X_train[y_train['Stroke'] == 1]
    X_train_no_stroke = X_train[y_train['Stroke'] == 2]
    X_train_no_stroke_random_selected = X_train_no_stroke.sample(number_of_stroke_sample)
    X_train_balanced = X_train_stroke.append(X_train_no_stroke_random_selected)
    y_train_stroke = y_train[y_train['Stroke'] == 1]
    y_train_no_stroke = y_train_stroke.replace({'Stroke':{1:2}})
    y_train_balanced = y_train_stroke.append(y_train_no_stroke)
    del y_train_balanced['Unnamed: 0']
    # save the selected and balanced data
    X_train_balanced.to_csv('./clean_data/train_X_balance.csv', index = False)
    X_test.to_csv('./clean_data/test_X_sel.csv', index = False)
    y_train_balanced.to_csv('./clean_data/train_y_balance.csv', index = False)
    print("Data Balancing Complete")
    return X_train_balanced, y_train_balanced, X_test


def import_clean_data():
    X_train = pd.read_csv('./clean_data/train_X_balance.csv')
    y_train = pd.read_csv('./clean_data/train_y_balance.csv')
    X_test = pd.read_csv('./clean_data/test_X_sel.csv')
    # X_train = X_train.sort_values(by=['Unnamed: 0'])
    # y_train = y_train.sort_values(by=['Unnamed: 0.1'])
    return X_train, y_train, X_test


def main():
    # X_train, y_train, X_test = import_original_data() # import original data
    X_train, y_train, X_test = import_clean_data()  # import clean data



if __name__ == '__main__':
    main()
