# coding=utf-8
"""
Graduate admission prediction kernel.


Dataset by:
Mohan S Acharya, Asfia Armaan, Aneeta S Antony :
A Comparison of Regression Models for Prediction of Graduate Admissions,
IEEE International Conference on Computational Intelligence in Data Science 2019
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn import linear_model


def load_dataset(path="Admission.csv", verbose=True):
    print("Loading data...")
    dataset = pd.read_csv(path)
    if verbose:
        print(dataset.head())
        print(dataset.describe())
    return dataset


if __name__ == "__main__":
    df = load_dataset()
    print(df)
    X = df.drop('Chance of Admit ', axis=1)
    y = df['Chance of Admit ']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    performance = []

    print("Linear regression")
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    score = r2_score(y_test, model.predict(X_test))
    print("R2 score: {0}".format(score))
    plt.scatter(np.arange(X_train.shape[0]), y_train, color="purple")
    plt.scatter(np.arange(X_test.shape[0]) + X_train.shape[0], y_test, color="red")
    plt.plot(np.arange(X_train.shape[0]), model.predict(X_train), color="purple")
    plt.plot(np.arange(X_test.shape[0]) + X_train.shape[0], model.predict(X_test), color="red")
    plt.show()
    performance.append(("Linear regression", score))

    print("LASSO")
    model = linear_model.Lasso()
    model.fit(X_train, y_train)
    score = r2_score(y_test, model.predict(X_test))
    print("R2 score: {0}".format(score))
    plt.scatter(np.arange(X_train.shape[0]), y_train, color="purple")
    plt.scatter(np.arange(X_test.shape[0]) + X_train.shape[0], y_test, color="red")
    plt.plot(np.arange(X_train.shape[0]), model.predict(X_train), color="purple")
    plt.plot(np.arange(X_test.shape[0]) + X_train.shape[0], model.predict(X_test), color="red")
    plt.show()
    performance.append(("LASSO", score))

    print("Ridge regression")
    model = linear_model.Ridge()
    model.fit(X_train, y_train)
    score = r2_score(y_test, model.predict(X_test))
    print("R2 score: {0}".format(score))
    plt.scatter(np.arange(X_train.shape[0]), y_train, color="purple")
    plt.scatter(np.arange(X_test.shape[0]) + X_train.shape[0], y_test, color="red")
    plt.plot(np.arange(X_train.shape[0]), model.predict(X_train), color="purple")
    plt.plot(np.arange(X_test.shape[0]) + X_train.shape[0], model.predict(X_test), color="red")
    plt.show()
    performance.append(("Ridge regression", score))

    print("LARS")
    model = linear_model.Lars()
    model.fit(X_train, y_train)
    score = r2_score(y_test, model.predict(X_test))
    print("R2 score: {0}".format(score))
    plt.scatter(np.arange(X_train.shape[0]), y_train, color="purple")
    plt.scatter(np.arange(X_test.shape[0]) + X_train.shape[0], y_test, color="red")
    plt.plot(np.arange(X_train.shape[0]), model.predict(X_train), color="purple")
    plt.plot(np.arange(X_test.shape[0]) + X_train.shape[0], model.predict(X_test), color="red")
    plt.show()
    performance.append(("LARS", score))

    print("Elastic net")
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    score = r2_score(y_test, model.predict(X_test))
    print("R2 score: {0}".format(score))
    plt.scatter(np.arange(X_train.shape[0]), y_train, color="purple")
    plt.scatter(np.arange(X_test.shape[0]) + X_train.shape[0], y_test, color="red")
    plt.plot(np.arange(X_train.shape[0]), model.predict(X_train), color="purple")
    plt.plot(np.arange(X_test.shape[0]) + X_train.shape[0], model.predict(X_test), color="red")
    plt.show()
    performance.append(("Elastic net", score))

    models = [result[0] for result in performance]
    scores = [result[1] for result in performance]
    x_pos = np.arange(len(models))

    plt.bar(x_pos, scores, align='center')
    plt.xticks(x_pos, models)
    plt.ylabel('Regression Score')
    plt.show()
