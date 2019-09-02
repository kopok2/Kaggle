# coding=utf-8
"""HTRU2 dataset.

High Time Resolution Universe Survey dataset kernel.

Features binary classification of star as pulsar based on 8 continuous attributes.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

# Parameters
VERBOSE = False


def load_dataset(path="pulsar_stars.csv", verbose=True):
    print("Loading data...")
    dataset = pd.read_csv(path)
    if verbose:
        print(dataset.head())
        print(dataset.describe())
    return dataset


def split_dataset(dataset):
    print("Spliting dataset...")
    X = dataset.drop("target_class", axis=1)
    y = dataset['target_class']
    return X, y


def eval_metric(confusion_mx):
    print(confusion_mx)
    base = confusion_mx[0][1] + sum(confusion_mx[1])
    base2 = sum(confusion_mx[0]) + sum(confusion_mx[1])
    good = confusion_mx[0][0] + confusion_mx[1][1]
    if base:
        print("KO score: {0:.2f}% | Accuracy: {1:.2f}%".format((confusion_mx[1][1] / base) * 100, (good / base2) * 100))
    else:
        print("KO score: 100.00% | Accuracy: 100.00%")


if __name__ == "__main__":
    df = load_dataset(verbose=VERBOSE)
    print(df)

    X, y = split_dataset(df)
    skf = StratifiedKFold(n_splits=5)
    for train_i, test_i in skf.split(X, y):
        oxtrain, oxtest = X.iloc[train_i], X.iloc[test_i]
        oytrain, oytest = y.iloc[train_i], y.iloc[test_i]
    X_train = oxtrain.values
    X_test = oxtest.values
    y_train = oytrain.values
    y_test = oytest.values

    print(X.describe())
    print(y.describe())
    print(X_train)
    print(X_test)
    print(y_train)
    print(y_test)
    print("LGBM model training...")
    train_data = lgb.Dataset(X_train, label=y_train)
    lr = 0.003
    param = {'objective': 'binary', "learning_rate": lr, 'metric': ['auc', 'accuracy', 'binary_logloss'], 'boosting': 'dart',
             'top_k': 2300, 'tree_learner': 'voting'}
    num_round = 7500

    lgb_model = lgb.train(param, train_data, num_round, valid_sets=[lgb.Dataset(X_test, y_test)],
                          early_stopping_rounds=10)
    print("Test")
    eval_metric(confusion_matrix(y_test, lgb_model.predict(X_test).round()))
    print("Training")
    eval_metric(confusion_matrix(y_train, lgb_model.predict(X_train).round()))

    print("Extra decision tree classifier")
    model = ExtraTreesClassifier(n_estimators=200, max_depth=None, min_samples_split=2)
    model.fit(X_train, y_train)
    print("Test")
    eval_metric(confusion_matrix(y_test, model.predict(X_test).round()))
    print("Training")
    eval_metric(confusion_matrix(y_train, model.predict(X_train).round()))

    print("Decision tree classifier")
    model = DecisionTreeClassifier(max_depth=None, min_samples_split=2)
    model.fit(X_train, y_train)
    print("Test")
    eval_metric(confusion_matrix(y_test, model.predict(X_test).round()))
    print("Training")
    eval_metric(confusion_matrix(y_train, model.predict(X_train).round()))

    print("Decision tree classifier with scaler and PCA")
    model = make_pipeline(StandardScaler(), PCA(n_components=4),
                          ExtraTreesClassifier(n_estimators=100, max_depth=None, min_samples_split=2))
    model.fit(X_train, y_train)
    print("Test")
    eval_metric(confusion_matrix(y_test, model.predict(X_test).round()))
    print("Training")
    eval_metric(confusion_matrix(y_train, model.predict(X_train).round()))

    print("Architecture ANN search")
    from Kaggle.ANN_constructor.ANNArchitectureGridSearch import architecture_grid_search, architecture_random_grid_search
    #architecture_random_grid_search(X_train, y_train, X_test, y_test)
    architecture_grid_search(X_train, y_train, X_test, y_test, max_width=12, max_depth=10)
