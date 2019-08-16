# coding=utf-8
"""New Credit Card Fraud Detection kernel.

Scaling and sub-sampling is being used.
"""

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical


# Hyperparameters
EPOCHS = 30
VERBOSE = 2
OPTIMIZER = SGD()
BATCH_SIZE = 256
VERBOSE = False


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


def construct_ANN_classifier(layers, width):
    print("Constructing neural network with {0} layers and {1} width...".format(layers, width))
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
    print("Credit Card Fraud Detection kernel.\nCopyright 2019 Karol Oleszek")

    df = load_dataset(VERBOSE)
    df = scale_dataset(df, VERBOSE)

    X_train, X_test, y_train, y_test = split_dataset(df)
    #y_test = to_categorical(y_test)
    #y_train = to_categorical(y_train)

    ndf = subsample_dataset(df, VERBOSE)
    undersampled_X_train = ndf.drop("Class", axis=1)
    undersampled_y_train = ndf["Class"]
    #undersampled_y_train = to_categorical(undersampled_y_train)

    ANN = construct_ANN_classifier(20, 72)
    ANN.compile(optimizer=OPTIMIZER, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print(undersampled_y_train)
    ANN.fit(undersampled_X_train, undersampled_y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2, validation_split=0.2)

    score = ANN.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)
    print(score)
    print(confusion_matrix(y_test, ANN.predict(X_test).argmax(axis=1)))
    #print(confusion_matrix(y_test.argmax(axis=1), ANN.predict(X_test).argmax(axis=1)))