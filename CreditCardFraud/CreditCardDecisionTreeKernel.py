# coding=utf-8
"""Credit Card Fraud Detection Decision Tree kernel."""


# coding=utf-8
"""New Credit Card Fraud Detection kernel.

Scaling and sub-sampling is being used.
"""

import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier


VERBOSE = True


def load_dataset(verbose=True):
    print("Loading data...")
    dataset = pd.read_csv("creditcard.csv")
    if verbose:
        print(dataset.head())
        print(dataset.describe())
    return dataset


def scale_dataset(dataset, verbose=True):
    print("Scaling data...")
    scaler = RobustScaler()
    dataset['scl_amount'] = scaler.fit_transform(dataset['Amount'].values.reshape(-1, 1))
    dataset['scl_time'] = scaler.fit_transform(dataset['Time'].values.reshape(-1, 1))
    dataset.drop(['Time', 'Amount'], axis=1, inplace=True)
    scl_amount = dataset['scl_amount']
    scl_time = dataset['scl_time']
    dataset.drop(['scl_amount', 'scl_time'], axis=1, inplace=True)
    dataset.insert(0, 'Amount', scl_amount)
    dataset.insert(0, 'Time', scl_time)
    if verbose:
        print(dataset.head())
        print(dataset.describe())
    return dataset


def split_dataset(dataset):
    skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    X = dataset.drop('Class', axis=1)
    y = dataset['Class']
    for train_i, test_i in skf.split(X, y):
        oxtrain, oxtest = X.iloc[train_i], X.iloc[test_i]
        oytrain, oytest = y.iloc[train_i], y.iloc[test_i]
    oxtrain = oxtrain.values
    oxtest = oxtest.values
    oytrain = oytrain.values
    oytest = oytest.values
    return oxtrain, oxtest, oytrain, oytest


def subsample_dataset(dataset, verbose=True):
    print("Subsampling dataset...")
    dataset = dataset.sample(frac=1)
    fraud_subsample = dataset.loc[dataset["Class"] == 1]
    valid_subsample = dataset.loc[dataset["Class"] == 0][:fraud_subsample.shape[0]]
    equal_subsample = pd.concat([fraud_subsample, valid_subsample])
    ndf = equal_subsample.sample(frac=1)
    if verbose:
        print(ndf.head())
        print(ndf.describe())
    return ndf


class DoubleTreeModel:
    def __init__(self, first_line_tree, second_line_tree):
        self.first_line_tree = first_line_tree
        self.second_line_tree = second_line_tree

    def predict(self, X):
        first_class = self.first_line_tree.predict(X)
        follow_up = X[first_class == 1]
        second_class = self.second_line_tree.predict(follow_up)
        result = np.zeros(len(X))
        y = 0
        for x in range(len(result)):
            if first_class[x]:
                if second_class[y]:
                    result[x] = 1
                y += 1
        return result


if __name__ == "__main__":
    print("Credit Card Fraud Detection kernel.\nCopyright 2019 Karol Oleszek")

    df = load_dataset(VERBOSE)
    df = scale_dataset(df, VERBOSE)

    X_train, X_test, y_train, y_test = split_dataset(df)

    ndf = subsample_dataset(df, VERBOSE)
    undersampled_X_train = ndf.drop("Class", axis=1)
    undersampled_y_train = ndf["Class"]

    for cr in ['gini', 'entropy']:
        for spl in ['best', 'random']:
            print(cr, spl)
            DCT = DecisionTreeClassifier(criterion=cr, splitter=spl, class_weight={0: 1, 1: 400})
            DCT.fit(undersampled_X_train, undersampled_y_train)

            if VERBOSE:
                print("First line")
                print("Test")
                print(confusion_matrix(y_test, DCT.predict(X_test)))
                print("Training")
                print(confusion_matrix(y_train, DCT.predict(X_train)))

            selected = DCT.predict(X_train)
            follow_X = X_train[selected == 1]
            follow_y = y_train[selected == 1]

            DCT_followup = DecisionTreeClassifier(criterion=cr, splitter=spl)
            DCT_followup.fit(follow_X, follow_y)

            if VERBOSE:
                print("Second line")
                print("Test")
                print(confusion_matrix(y_test, DCT_followup.predict(X_test)))
                print("Training")
                print(confusion_matrix(follow_y, DCT_followup.predict(follow_X)))

            DOUBLE_MODEL = DoubleTreeModel(DCT, DCT_followup)
            print("Double model")
            print("Test")
            print(confusion_matrix(y_test, DOUBLE_MODEL.predict(X_test)))
            print("Training")
            print(confusion_matrix(y_train, DOUBLE_MODEL.predict(X_train)))
