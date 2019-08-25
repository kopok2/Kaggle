# coding=utf-8
"""Heart Disease UCI dataset kernel.

Heart disease prediction model.
"""

import pandas as pd
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


def load_dataset(path="heart.csv", verbose=True):
    print("Loading data...")
    dataset = pd.read_csv(path)
    if verbose:
        print(dataset.head())
        print(dataset.describe())
    return dataset


def split_dataset(dataset):
    print("Spliting dataset...")
    skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    X = dataset.drop('target', axis=1)
    y = dataset['target']
    for train_i, test_i in skf.split(X, y):
        oxtrain, oxtest = X.iloc[train_i], X.iloc[test_i]
        oytrain, oytest = y.iloc[train_i], y.iloc[test_i]
    oxtrain = oxtrain.values
    oxtest = oxtest.values
    oytrain = oytrain.values
    oytest = oytest.values
    return oxtrain, oxtest, oytrain, oytest


def eval_metric(confusion_mx):
    print(confusion_mx)
    base = confusion_mx[0][1] + sum(confusion_mx[1])
    if base:
        print("{0:.2f}%".format((confusion_mx[1][1] / base) * 100))
    else:
        print("100.00%")


if __name__ == "__main__":
    df = load_dataset(verbose=VERBOSE)
    X_train, X_test, y_train, y_test = split_dataset(df)
    #grid = architecture_grid_search(X_train, y_train, X_test, y_test)
    print("LGBM model training...")
    train_data = lgb.Dataset(X_train, label=y_train)
    lr = 0.98169
    #lr = 1
    param = {'objective': 'binary', "learning_rate": lr, 'metric': ['auc', 'accuracy', 'binary_logloss'], 'boosting': 'dart',
             'bagging_freq': 3, 'bagging_fraction': 0.75, 'top_k': 1000, 'tree_learner': 'voting'}
    num_round = 100
    lgb_model = lgb.train(param, train_data, num_round, valid_sets=[lgb.Dataset(X_test, y_test)],
                          early_stopping_rounds=1)
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
    model = make_pipeline(StandardScaler(), PCA(n_components=7),
                          ExtraTreesClassifier(n_estimators=225, max_depth=None, min_samples_split=2))
    model.fit(X_train, y_train)
    print("Test")
    eval_metric(confusion_matrix(y_test, model.predict(X_test).round()))
    print("Training")
    eval_metric(confusion_matrix(y_train, model.predict(X_train).round()))
