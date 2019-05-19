from imblearn.over_sampling import SMOTE
from scipy.io import arff
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model.logistic import LogisticRegression
from xgboost import XGBClassifier

import numpy as np

def random_forest(X_train,train_target, X_test):
    rf = RandomForestClassifier()
    rf.fit(X_train, train_target)
    hyp = rf.predict(X_test)
    return hyp


def decision_tree(X_train,train_target, X_test):
    clf = DecisionTreeClassifier()
    clf.fit(X_train, train_target)
    hyp = clf.predict(X_test)
    return hyp


def SGD_classifier(X_train,train_target, X_test):
    clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
    clf.fit(X_train, train_target)
    hyp = clf.predict(X_test)
    return hyp

def Logistic_Regression_classifier(X_train, train_target, X_test):
    lr = LogisticRegression(solver='lbfgs').fit(X_train, train_target)
    return lr.predict(X_test)

def Knn(X_train, train_target, X_test):
    return KNeighborsClassifier().fit(X_train, train_target).predict(X_test)


def load_df(path):
    data = arff.loadarff(path)
    df = pd.DataFrame(data[0])
    df_target = df.iloc[:, 0]
    df_X = df.iloc[:, 1:]
    train_target = encode_y(df_target)
    return df_X,train_target


def encode_y(y_train):
    le = LabelEncoder()
    le.fit(y_train)
    train_target = le.transform(y_train)
    return train_target


def smoth_sampling(X_train,train_target):
    sm = SMOTE()
    x_train_over, y_train_over = sm.fit_sample(X_train, train_target)
    print('Origin dataset shape %s' % Counter(train_target))
    print('Resampled dataset shape by SMOTE  %s' % Counter(y_train_over))
    return x_train_over, y_train_over


def main():
    paths = ['../dataset/models_Yifan/PONZI_100000.arff','../dataset/models_Yifan/PONZI_basic_100000.arff','../dataset/models_Yifan/PONZI_opcodes_100000.arff']
    for path in paths:
        print('\n'+ path +'\n')
        df_X, df_target = load_df(path)

        X_train, X_test, y_train, y_test = train_test_split(df_X, df_target, test_size=0.33, random_state=42)
        x_train_over, y_train_over = smoth_sampling(X_train, y_train)
        hyp = Knn(x_train_over,y_train_over, X_test)
        print('Accuracy(Resampled KNN) is %s' % accuracy_score(hyp, y_test))
        print('Confusion matrix:\n %s' % confusion_matrix(hyp, y_test))

        hyp = random_forest(x_train_over,y_train_over, X_test)
        print('Accuracy(Resampled RandomForest) is %s' % accuracy_score(hyp, y_test))
        print('Confusion matrix:\n %s' % confusion_matrix(hyp, y_test))

        hyp = SGD_classifier(x_train_over, y_train_over, X_test)
        print('Accuracy(Resampled SGDClassifier) is %s' % accuracy_score(hyp, y_test))
        print('Confusion matrix:\n %s' % confusion_matrix(hyp, y_test))

        hyp = Logistic_Regression_classifier(x_train_over, y_train_over, X_test)
        print('Accuracy(Resampled Logistic Regression Classifier) is %s' % accuracy_score(hyp, y_test))
        print('Confusion matrix:\n %s' % confusion_matrix(hyp, y_test))

        # knn = KNeighborsClassifier()
        # scores = cross_val_score(knn, df_X, df_target, cv=10)
        # print('Knn accuracy is %s' %np.average(scores))
        #
        # rf = RandomForestClassifier(n_estimators=100)
        # scores = cross_val_score(rf, df_X, df_target, cv=10)
        # print('Random Forest Classifier accuracy is %s' %np.average(scores))
        #
        # sgd = SGDClassifier(loss="hinge", penalty="l2", tol=1e-3, max_iter=10000)
        # scores = cross_val_score(sgd, df_X, df_target, cv=10)
        # print('SGDClassifier accuracy is %s' %np.average(scores))
        #
        # lr = LogisticRegression(solver='lbfgs')
        # scores = cross_val_score(lr, df_X, df_target, cv=10)
        # print('Logistic Regression accuracy is %s' %np.average(scores))



main()
