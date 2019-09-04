# coding=utf-8
"""SMS Spam Detection Kernel."""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer

# Parameters
VERBOSE = False


def load_dataset(path="spam.csv", verbose=True):
    print("Loading data...")
    dataset = pd.read_csv(path, dtype=str)
    if verbose:
        print(dataset.head())
        print(dataset.describe())
    return dataset


def split_dataset(dataset):
    print("Spliting dataset...")
    X = dataset['v2']
    y = dataset['v1']
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

    X, y= split_dataset(df)
    print(X)
    print(y)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X)
    X = pd.DataFrame(X.toarray())
    y = pd.DataFrame(OneHotEncoder().fit_transform(np.array(y).reshape(-1, 1)).toarray()).loc[:, 1]

    skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
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

    print("\n" + "#" * 128)
    print("LGBM model training...")
    train_data = lgb.Dataset(X_train, label=y_train)
    lr = 0.01
    param = {'objective': 'binary', "learning_rate": lr, 'metric': ['auc', 'accuracy', 'binary_logloss'], 'boosting': 'dart',
             'top_k': 2300, 'tree_learner': 'voting'}
    num_round = 1000

    lgb_model = lgb.train(param, train_data, num_round, valid_sets=[lgb.Dataset(X_test, y_test)],
                          early_stopping_rounds=10)
    print("Test")
    eval_metric(confusion_matrix(y_test, lgb_model.predict(X_test).round()))
    print("Training")
    eval_metric(confusion_matrix(y_train, lgb_model.predict(X_train).round()))

    print("\n" + "#" * 128)
    print("Extra decision tree classifier")
    model = ExtraTreesClassifier(n_estimators=200, max_depth=None, min_samples_split=2)
    model.fit(X_train, y_train)
    print("Test")
    eval_metric(confusion_matrix(y_test, model.predict(X_test).round()))
    print("Training")
    eval_metric(confusion_matrix(y_train, model.predict(X_train).round()))

    print("\n" + "#" * 128)
    print("Decision tree classifier")
    model = DecisionTreeClassifier(max_depth=None, min_samples_split=2)
    model.fit(X_train, y_train)
    print("Test")
    eval_metric(confusion_matrix(y_test, model.predict(X_test).round()))
    print("Training")
    eval_metric(confusion_matrix(y_train, model.predict(X_train).round()))

    print("\n" + "#" * 128)
    print("Decision tree classifier with scaler and PCA")
    model = make_pipeline(StandardScaler(), PCA(n_components=13),
                          ExtraTreesClassifier(n_estimators=225, max_depth=None, min_samples_split=2))
    model.fit(X_train, y_train)
    print("Test")
    eval_metric(confusion_matrix(y_test, model.predict(X_test).round()))
    print("Training")
    eval_metric(confusion_matrix(y_train, model.predict(X_train).round()))

    print("\n" + "#" * 128)
    print("Naive Bayes classifier with scaler and PCA")
    model = make_pipeline(StandardScaler(), PCA(n_components=20),
                          GaussianNB())
    model.fit(X_train, y_train)
    print("Test")
    eval_metric(confusion_matrix(y_test, model.predict(X_test).round()))
    print("Training")
    eval_metric(confusion_matrix(y_train, model.predict(X_train).round()))

    print("\n" + "#" * 128)
    print("Naive Bayes Classifier")
    model = GaussianNB()
    model.fit(X_train, y_train)
    print("Test")
    eval_metric(confusion_matrix(y_test, model.predict(X_test).round()))
    print("Training")
    eval_metric(confusion_matrix(y_train, model.predict(X_train).round()))

    print("\n" + "#" * 128)
    print("Architecture ANN search")
    from Kaggle.ANN_constructor.ANNArchitectureGridSearch import architecture_grid_search, architecture_random_grid_search
    architecture_random_grid_search(X_train, y_train, X_test, y_test)
    architecture_grid_search(X_train, y_train, X_test, y_test)
