# Course project source code
# Copyright @ Siqi Ping (2022214047) Zihan Wang (2022312876)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve


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
    X_train = pd.read_csv(
        '/Users/zihan/Documents/Learning/研究生课程/ML2022/Project/Stroke_prediction_system_Data/Prf_feature_train.csv')
    X_test = pd.read_csv(
        '/Users/zihan/Documents/Learning/研究生课程/ML2022/Project/Stroke_prediction_system_Data/Prf_feature_test.csv')
    train_label = pd.read_csv(
        '/Users/zihan/Documents/Learning/研究生课程/ML2022/Project/Stroke_prediction_system_Data/Stroke_label_train.csv')
    # Select relavent features
    selected_features = pd.read_csv('./Feature_selected.csv')
    # Manully select features
    # X_train = X_train[selected_features['Feature']]
    # X_test = test_data[selected_features['Feature']]
    # Only researve data with full record
    y_train = train_label[X_train['DISPCODE'] == 1100]
    X_train = X_train[X_train['DISPCODE'] == 1100]
    # X_test = X_test[X_test['DISPCODE'] == 1100]
    # 删除数据记录完整度的栏目
    # del X_test['DISPCODE']
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
    X_train_balanced.to_csv('./data/train_X_balance.csv', index = False)
    # X_test.to_csv('./clean_data/test_X_sel.csv', index = False)
    y_train_balanced.to_csv('./data/train_y_balance.csv', index = False)
    print("Data Balancing Complete")
    return X_train_balanced, y_train_balanced, X_test


def import_clean_data():
    X_train = pd.read_csv('./data/train_X_balance.csv')
    y_train = pd.read_csv('./data/train_y_balance.csv')
    X_test = pd.read_csv('./data/Prf_feature_test.csv')
    # X_train = X_train.sort_values(by=['Unnamed: 0'])
    # y_train = y_train.sort_values(by=['Unnamed: 0.1'])
    return X_train, y_train, X_test


def plot_errs(train_sizes, train_scores, validation_scores, model_name):
    train_scores_mean = -train_scores.mean(axis=1)
    validation_scores_mean = -validation_scores.mean(axis=1)
    print('Mean training scores for ' + model_name + '\n', pd.Series(train_scores_mean, index=train_sizes))
    print('\n', '-' * 20)  # separator
    print('\nMean validation scores ' + model_name + '\n', pd.Series(validation_scores_mean, index=train_sizes))
    plt.style.use('seaborn')
    plt.plot(train_sizes, train_scores_mean, label='Training error')
    plt.plot(train_sizes, validation_scores_mean, label='Validation error')
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    plt.title('Learning curves for a ' + model_name + ' model', fontsize=18, y=1.03)
    plt.legend()
    plt.show()


def prediction_results(pred, y_test, name_of_classifier):
    tn, fp, fn, tp = confusion_matrix(list(y_test), list(pred)).ravel()
    print("Results by Scikit-Learn library of", name_of_classifier)
    print('True Positive', tp)
    print('True Negative', tn)
    print('False Positive', fp)
    print('False Negative', fn)
    ConfusionMatrixDisplay.from_predictions(y_test, pred)
    plt.title(name_of_classifier)
    plt.show()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = float(2 * precision * recall) / float(precision + recall)
    accuracy = float(tp + tn) / float(tp + tn + fp + fn)
    print("precision", precision, "\nrecall", recall, "\nf_score", f_score, "\naccuracy", accuracy)


def plot_ROC(FPR, TPR, title, area):
    fig, ax = plt.subplots()
    ax.plot(FPR, TPR, label='ROC curve (area = %0.2f)' % area)
    ax.plot((0, 1), (0, 1), ':', color='grey')
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect('equal')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    plt.title(title)
    plt.show()


def main():
    # X_train, y_train, X_test = import_original_data() # import original data
    X_train, y_train, X_test = import_clean_data()  # import clean data
    # Use decision tree to sort feature importance
    # Drop features with >90% NaN
    X_train = X_train.dropna(axis='columns',thresh=0.1*X_train.shape[0])
    # fill the NaN with mean values
    X_train_fill = X_train.fillna(X_train.mean())
    tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
    tree.fit(X_train_fill, y_train)
    # print(X_train_fill.columns, tree.feature_importances_)
    feature_importances = pd.concat([pd.DataFrame(X_train_fill.columns, columns=['features']),
                                     pd.DataFrame(tree.feature_importances_, columns=['importance'])], axis=1)
    feature_importances = feature_importances.sort_values(by='importance', ascending=False)
    top5features = feature_importances['features'][0:5]
    # indexing the most representative features
    X_test = X_test[top5features]
    X_train = X_train[top5features]
    # Split validation set from training set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=1,stratify=y_train)
    # Use logistic regression to predict Stroke onset
    lr = LogisticRegression()
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator=lr, X=X_train, y=y_train['Stroke'], train_sizes=[100, 500, 1000, 2000, 4000],
        cv=5, scoring='neg_mean_squared_error')
    plot_errs(train_sizes, train_scores, validation_scores, 'Logistic Regression')
    lr.fit(X_train, y_train['Stroke'])
    prediction_results(lr.predict(X_val), y_val['Stroke'], "Logistic Regression")
    # plot ROC curve
    y_pred_proba = lr.predict_proba(X_val)[::, 1]
    fpr, tpr, _ = roc_curve(y_val['Stroke'], y_pred_proba, pos_label=2)
    area = roc_auc_score(y_val['Stroke'], y_pred_proba)
    plot_ROC(fpr, tpr, 'Test for LogisticRegression', area)
    y_test = lr.predict(X_test)
    y_test = pd.DataFrame(y_test, columns=['prediction result'])
    y_test.to_csv('Test Set Prediction Result.csv')


if __name__ == '__main__':
    main()
