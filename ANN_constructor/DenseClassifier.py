# coding=utf-8
"""Universal Sklearn compatible ANN classifier model constructor."""

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam, RMSprop


class ANNClassifier:
    def __init__(self, input_shape, output_shape, architecture=[(3, 16)], EPOCHS=150, VERBOSE=2, OPTIMIZER=Adam,
                 BATCH_SIZE=256, regularize=False, metrics=["accuracy"], loss="mean_squared_error"):
        # Hyperparameters
        self.EPOCHS = EPOCHS
        self.VERBOSE = VERBOSE
        self.OPTIMIZER = OPTIMIZER()
        self.BATCH_SIZE = BATCH_SIZE
        model = Sequential()
        model.add(Dense(architecture[0][1], input_shape=(input_shape,)))
        model.add(Activation("relu"))
        for part in architecture:
            for layer in range(part[0]):
                model.add(Dense(part[1]))
                model.add(Activation("relu"))
                if regularize:
                    model.add(Dropout(0.2))
        model.add(Dense(output_shape))
        model.add(Activation("softmax"))
        self.model = model
        self.model.compile(optimizer=self.OPTIMIZER, loss=loss, metrics=metrics)
        if VERBOSE:
            print(self.model.summary())

    def fit(self, X, y):
        self.model.fit(X, y, batch_size=self.BATCH_SIZE, epochs=self.EPOCHS, verbose=self.VERBOSE, validation_split=0.2)

    def predict(self, X):
        return self.model.predict(X)
