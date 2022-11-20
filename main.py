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



def main():
    # import data
    train_data = pd.read_csv('./Stroke_prediction_system_Data/Prf_feature_train.csv')
    train_label = pd.read_csv('./Stroke_prediction_system_Data/Prf_feature_test.csv')
    Stroke_label_train = pd.read_csv('./Stroke_prediction_system_Data/Stroke_label_train.csv')
    print("Load data complete")

if __name__ == '__main__':
    main()
