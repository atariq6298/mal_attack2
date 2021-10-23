import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers
import load_dataset


class Rnn:
    def __init__(self, dataset):
        X_train, X_test, y_train, y_test = dataset
        # X_train = X_train.to_numpy().reshape(-1, 1, 16)
        # X_test = X_test.to_numpy().reshape(-1, 1, 16)
        # y_train = y_train.reshape(-1, 1, 2)
        # y_test = y_test.reshape(-1, 1, 2)

        model = keras.Sequential()

        # Add a LSTM layer with 128 internal units.
        model.add(layers.LSTM(128))

        # Add a Dense layer with 10 units.
        model.add(layers.Dense(2, activation='softmax'))

        model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)

        # print(model.predict(X_test[:4]))
        prediction = np.argmax(model.predict(X_test), axis=1)
        y_test = np.argmax(y_test.reshape(2200, 2), axis=1)
        print("RNN Results Start ---------------------------")
        print('\n Accuracy: ')
        print(accuracy_score(y_test, prediction))
        print('\n F1 score: ')
        print(f1_score(y_test, prediction))
        print('\n Recall: ')
        print(recall_score(y_test, prediction))
        print('\n Precision: ')
        print(precision_score(y_test, prediction))
        print('\n confusion matrix: \n')
        print(confusion_matrix(y_test, prediction))
        print("RNN Results End ---------------------------")
        self.model = model
        self.prediction = prediction
        self.prob = model.predict(X_test)