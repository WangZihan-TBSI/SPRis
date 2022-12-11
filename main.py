# Course project source code
# Copyright @ Siqi Ping (2022214047) Zihan Wang (2022312876)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer


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


def import_data():
    train_data = pd.read_csv(
        '/Users/zihan/Seafile/Learning/研究生课程/ML2022/Project/Stroke_prediction_system_Data/Prf_feature_train.csv')
    test_data = pd.read_csv(
        '/Users/zihan/Seafile/Learning/研究生课程/ML2022/Project/Stroke_prediction_system_Data/Prf_feature_test.csv')
    train_label = pd.read_csv(
        '/Users/zihan/Seafile/Learning/研究生课程/ML2022/Project/Stroke_prediction_system_Data/Stroke_label_train.csv')
    print("Data Loading Complete")
    # all_features = list(train_data.columns)
    # Select relavent features
    selected_features = pd.read_csv('./Feature_selected.csv')
    # 筛选使用数据
    X_train = train_data[selected_features['Feature']]
    X_test = train_data[selected_features['Feature']]
    X_train = X_train[X_train['DISPCODE'] == 1100]
    X_test = X_test[X_train['DISPCODE'] == 1100]
    del X_test['DISPCODE']
    del X_train['DISPCODE']
    return X_train, train_label, X_test


def main():
    # import data
    X_train, y_train, X_test = import_data()


if __name__ == '__main__':
    main()
