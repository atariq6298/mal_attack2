import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import load_dataset

class Ann:
    def __init__(self, dataset):
        X_train, X_test, y_train, y_test = dataset

        model = Sequential()
        model.add(Dense(16, activation = 'relu',))
        model.add(Dense(2, activation = 'relu',))
        model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)
        print(model.predict(X_test[:4]))
        prediction = np.argmax(model.predict(X_test), axis=1)
        y_test = np.argmax(y_test.reshape(2200, 2), axis=1)  # 2200 original

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
        self.model = model
        self.prediction = prediction