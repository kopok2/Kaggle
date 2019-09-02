# coding=utf-8
"""ANN Dense Classifier architecture random grid search."""

from random import randrange
from operator import itemgetter
import numpy as np
from keras.optimizers import SGD, Adam, RMSprop
from Kaggle.ANN_constructor.DenseClassifier import ANNClassifier
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical


def random_architecture(max_depth, max_width, min_width):
    layers_remaining = max_depth
    architecture = []
    while layers_remaining > 1:
        layering = randrange(1, layers_remaining)
        layers_remaining -= layering
        architecture.append((layering, randrange(min_width, max_width)))
    return architecture


def fixed_architecture(depth, width):
    return [(depth, width)]


def confusion_metric(confusion_mx):
    correct = 0
    for row in confusion_mx:
        print(row)
    base = sum([sum(row) for row in confusion_mx])
    for x in range(len(confusion_mx)):
        correct += confusion_mx[x][x]
    if not base:
        return 100.0
    else:
        return (correct / base) * 100


def architecture_random_grid_search(X, y, X_val, y_val, iterations=50, max_depth=256,
                             max_width=256, min_width=2, max_epochs=256):
    results = []
    for d_param in range(3, max_width + max_depth):
        architecture = random_architecture(d_param, d_param, min_width)
        optimizer = [SGD, Adam, RMSprop][randrange(3)]
        regularize = [True, False][randrange(2)]
        epochs = randrange(20, max_epochs)
        loss = ["categorical_crossentropy", "mean_squared_error",
                "mean_absolute_error", "mean_squared_logarithmic_error"][randrange(4)]

        model = ANNClassifier(X.shape[1], len(np.unique(y)),
                              architecture=architecture,
                              VERBOSE=0, OPTIMIZER=optimizer,
                              regularize=regularize, EPOCHS=epochs, loss=loss)
        y_train = to_categorical(y)
        model.fit(X, y_train)
        result = confusion_metric(confusion_matrix(y_val, model.predict(X_val).argmax(axis=1)))
        print(result, architecture, optimizer, regularize, epochs, loss)
        results.append((result, architecture, optimizer, regularize, epochs, loss))
    results.sort(key=itemgetter(0), reverse=True)
    for r in results:
        print(r)
    return results


def architecture_grid_search(X, y, X_val, y_val, max_depth=40,
                             max_width=40, min_width=2, max_epochs=256):
    results = []

    for optimizer in [Adam, RMSprop, SGD]:
        for regularize in [True, False]:
            for loss in ["mean_squared_logarithmic_error", "categorical_crossentropy", "mean_squared_error",
                         "mean_absolute_error"]:
                for x in range(1, max_depth):
                    for yp in range(1, max_width):

                        architecture = fixed_architecture(x, yp)
                        epochs = (x + yp) * 6
                        model = ANNClassifier(X.shape[1], len(np.unique(y)),
                                              architecture=architecture,
                                              VERBOSE=0, OPTIMIZER=optimizer,
                                              regularize=regularize, EPOCHS=epochs, loss=loss)
                        y_train = to_categorical(y)
                        model.fit(X, y_train)
                        result = confusion_metric(confusion_matrix(y_val, model.predict(X_val).argmax(axis=1)))
                        train_result = confusion_metric(confusion_matrix(y, model.predict(X).argmax(axis=1)))
                        print(result, train_result, architecture, optimizer, regularize, epochs, loss)
                        results.append((result, train_result, architecture, optimizer, regularize, epochs, loss))
    results.sort(key=itemgetter(0), reverse=True)
    for r in results:
        print(r)
    return results
