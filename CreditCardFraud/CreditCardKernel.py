# coding=utf-8
"""Credit card fraud detection kernel."""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical
from keras import backend as K
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

# Hyperparameters
EPOCHS = 10
VERBOSE = 2
OPTIMIZER = SGD()
BATCH_SIZE = 256


def normalize(df):
    x = df.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    return pd.DataFrame(x_scaled)


def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


def construct_ANN_classifier(layers, width):
    model = Sequential()
    model.add(Dense(width, input_shape=(30,)))
    model.add(Activation("relu"))
    for layer in range(layers):
        model.add(Dense(width))
        model.add(Activation("relu"))
    model.add(Dense(2))
    model.add(Activation("softmax"))
    return model


if __name__ == "__main__":
    print("Loading data...")
    data = pd.read_csv("creditcard.csv")
    features = data.loc[:, data.columns != "Class"]
    X_train = features.sample(frac=0.8, random_state=200)
    X_test = features.drop(X_train.index)
    targets = data.loc[:, data.columns == "Class"]
    y_train = targets.loc[X_train.index]
    y_test = targets.drop(X_train.index)
    # copy sparse data to train set
    """aug = data.loc[X_train.index]
    aug = aug[data["Class"] == 1]
    aug_X = pd.concat([aug.loc[:, aug.columns != "Class"]] * 500)
    aug_y = pd.concat([aug.loc[:, aug.columns == "Class"]] * 500)
    X_train = X_train.append(aug_X, ignore_index=True)
    y_train = y_train.append(aug_y, ignore_index=True)"""
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    #print(aug_X, aug_y)

    print(X_train, y_train, X_test, y_test)
    ANN = construct_ANN_classifier(20, 128)
    ANN.summary()

    print(y_test)
    class_weights = {0: 1, 1: 300}
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    print(class_weights)
    ANN.compile(optimizer=OPTIMIZER, loss="categorical_crossentropy", metrics=[specificity, "accuracy"])
    ANN.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE, validation_split=0.2, class_weight=class_weights)

    score = ANN.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)
    print(score)
    print(confusion_matrix(y_test.argmax(axis=1), ANN.predict(X_test).argmax(axis=1)))
