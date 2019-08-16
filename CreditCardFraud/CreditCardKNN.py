# coding=utf-8
"""Credit card fraud detection kernel."""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing


def normalize(df):
    x = df.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    return pd.DataFrame(x_scaled)


if __name__ == "__main__":
    print("Loading data...")
    data = pd.read_csv("creditcard.csv")
    features = data.loc[:, data.columns != "Class"]
    X_train = features.sample(frac=0.2, random_state=200)
    X_test = features.drop(X_train.index)
    targets = data.loc[:, data.columns == "Class"]
    y_train = targets.loc[X_train.index].values
    y_train = y_train.reshape(len(y_train), 1)
    y_test = targets.drop(X_train.index).values
    y_test = y_test.reshape(len(y_test), 1)

    print("KNN classifier")
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X_train, y_train)
    print(confusion_matrix(y_test, knn.predict(X_test)))

    print("Normalizing data...")
    X_train = normalize(X_train)
    X_test = normalize(X_test)


    print(X_train, y_train, X_test, y_test)
    print("Nearest Centroid model:")
    model = NearestCentroid(metric='euclidean')
    model.fit(X_train, y_train)
    print(confusion_matrix(y_test, model.predict(X_test)))

    for x in range(1, 15):
        for w in ['uniform', 'distance']:
            for a in ['ball_tree', 'kd_tree', 'auto']:
                print("Neighbours: {0} Weights: {1} Algorithm: {2}".format(x, w, a))
                model = KNeighborsClassifier(n_neighbors=x, weights=w, algorithm=a)
                model.fit(X_train, y_train)
                print(confusion_matrix(y_test, model.predict(X_test)))

