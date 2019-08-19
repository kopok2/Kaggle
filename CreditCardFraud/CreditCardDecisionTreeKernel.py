# coding=utf-8
"""Credit Card Fraud Detection Decision Tree kernel."""


# coding=utf-8
"""New Credit Card Fraud Detection kernel.

Scaling and sub-sampling is being used.
"""

import numpy as np
from sklearn.preprocessing import RobustScaler, LabelBinarizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb
import lightgbm as lgb
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical
import keras.backend as K
import tensorflow as tf
# Hyperparameters
EPOCHS = 100
OPTIMIZER = Adam(lr=0.0002)
BATCH_SIZE = 512
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
                try:
                    if second_class[y]:
                        result[x] = 1
                except:
                    if second_class[y][0] < second_class[y][1]:
                        result[x] = 1
                y += 1
        return result


def eval_metric(confusion_mx):
    print(confusion_mx)
    base = confusion_mx[0][1] + sum(confusion_mx[1])
    if base:
        print("{0:.2f}%".format((confusion_mx[1][1] / base) * 100))
    else:
        print("100.00%")


def construct_ANN_classifier(layers, width):
    print("Constructing neural network with {0} layers and {1} width...".format(layers, width))
    model = Sequential()
    model.add(Dense(width, input_shape=(30,)))
    model.add(Activation("sigmoid"))
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

    ndf = subsample_dataset(df, VERBOSE)
    undersampled_X_train = ndf.drop("Class", axis=1)
    undersampled_y_train = ndf["Class"]

    print("LGBM model training...")
    train_data = lgb.Dataset(X_train, label=y_train)
    param = {'objective': 'binary', "learning_rate":0.01}
    param['metric'] = ['auc']
    num_round = 500
    lgb_model = lgb.train(param, train_data, num_round, valid_sets=[lgb.Dataset(X_test, y_test)])
    print("Test")
    eval_metric(confusion_matrix(y_test, lgb_model.predict(X_test).round()))
    print("Training")
    eval_metric(confusion_matrix(y_train, lgb_model.predict(X_train).round()))

    for cr in ['gini']: #, 'entropy']:
        for spl in ['best']: #, 'random']:
            second_line_models = [DecisionTreeClassifier(criterion=cr, splitter=spl),
                                  KNeighborsClassifier(n_neighbors=4),
                                  NearestCentroid(metric='euclidean'),
                                  SVC(gamma="auto", class_weight="balanced"),
                                  #MLPClassifier(alpha=1, max_iter=1000),
                                  #AdaBoostClassifier(),
                                  #RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                                  #SGDClassifier(max_iter=1000, tol=1e-3),
                                  #GaussianNB(),
                                  #QuadraticDiscriminantAnalysis(),
                                  #GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                  #                           max_depth=1),
                                  ExtraTreesClassifier(n_estimators=50, max_depth=None,
                                                       min_samples_split=2, criterion=cr),
                                  #xgb
                                  lgb,
                                  "ann"
                                  ]
            for slm in second_line_models:
                print(cr, spl, slm)
                DCT = DecisionTreeClassifier(criterion=cr, splitter=spl, class_weight={0: 1, 1: 400})
                DCT.fit(undersampled_X_train, undersampled_y_train)

                if VERBOSE:
                    print("First line")
                    print("Test")
                    eval_metric(confusion_matrix(y_test, DCT.predict(X_test)))
                    print("Training")
                    eval_metric(confusion_matrix(y_train, DCT.predict(X_train)))

                selected = DCT.predict(X_train)
                follow_X = X_train[selected == 1]
                follow_y = y_train[selected == 1]

                DCT_followup = slm
                if "xgboost" in str(slm):
                    DCT_followup = xgb.XGBClassifier()
                    DCT_followup.set_params(**{'predictor': 'cpu_predictor', 'max_depth':2, 'eta':1, 'objective':'binary:logistic'})
                    DCT_followup.fit(follow_X, follow_y)
                elif "lightgbm" in str(slm):
                    train_data = lgb.Dataset(follow_X, label=follow_y)
                    param = {'objective': 'binary', "learning_rate":0.01}
                    param['metric'] = ['binary_error']
                    num_round = 500
                    DCT_followup = lgb.train(param, train_data, num_round)
                elif "ann" in str(slm): # 20 72
                    DCT_followup = construct_ANN_classifier(25, 80)
                    DCT_followup.summary()
                    DCT_followup.compile(optimizer=OPTIMIZER, loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
                    print(undersampled_y_train)
                    DCT_followup.fit(follow_X, follow_y, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2,
                            validation_split=0.2)
                else:
                    DCT_followup.fit(follow_X, follow_y)

                if VERBOSE:
                    print("Second line")
                    print("Test")
                    eval_metric(confusion_matrix(y_test, DCT_followup.predict(X_test)))
                    print("Training")
                    eval_metric(confusion_matrix(follow_y, DCT_followup.predict(follow_X)))

                DOUBLE_MODEL = DoubleTreeModel(DCT, DCT_followup)
                print("Double model")
                print("Test")
                eval_metric(confusion_matrix(y_test, DOUBLE_MODEL.predict(X_test).round()))
                print("Training")
                eval_metric(confusion_matrix(y_train, DOUBLE_MODEL.predict(X_train).round()))
