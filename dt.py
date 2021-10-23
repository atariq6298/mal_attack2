# Create Decision Tree classifer object
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

import load_dataset


class Dt:
    def __init__(self, dataset):
        X_train, X_test, y_train, y_test = dataset

        model = DecisionTreeClassifier()

        # Train Decision Tree Classifer
        model = model.fit(X_train, y_train)

        # Predict the response for test dataset
        y_pred = model.predict(X_test)

        # show predictions for the first 3 images in the test set
        # model.predict(X_test[:4])
        prediction = np.argmax(model.predict(X_test), axis=1)
        y_test = np.argmax(y_test, axis=1)
        print("DT Results Start ---------------------------")
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
        print("DT Results End ---------------------------")
        self.model = model
        self.prediction = prediction
        self.prob = model.predict(X_test)
