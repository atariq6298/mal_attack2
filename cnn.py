from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

import load_dataset
import numpy as np


class Cnn:

    # download mnist data and split into train and test sets
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()
    def __init__(self, dataset):
        X_train, X_test, y_train, y_test = dataset

        # plot the first image in the dataset
        #plt.imshow(X_train[0])
        train_rows, train_x, train_y = X_train.shape

        text_rows, test_x, test_y = X_test.shape

        # reshape data to fit model
        X_train = X_train.reshape(train_rows, train_x, train_y, 1)
        X_test = X_test.reshape(text_rows, test_x, test_y, 1)

        # one-hot encode target column
        # y_train = to_categorical(y_train)
        # y_test = to_categorical(y_test)

        # create model
        model = Sequential()

        # add model layers
        model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(train_x, train_y,
                                                                            1)))  # using 4 x 4 data as input hence donot need maxpooling layer which decrease the dimensionality of data by taking maximum value piont from each quater.make 2 x 2 array from 4 x 4.
        model.add(Conv2D(32, kernel_size=2, activation='relu'))
        model.add(Flatten())
        model.add(Dense(2, activation='softmax'))

        # compile model using accuracy as a measure of model performance
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # train model
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)

        # show predictions for the first 3 images in the test set
        # model.predict(X_test[:4])
        prediction = np.argmax(model.predict(X_test), axis=1)
        y_test = np.argmax(y_test, axis=1)
        print("CNN Results Start ---------------------------")
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
        print("CNN Results End ---------------------------")
        self.model = model
        self.prediction = prediction
        self.prob = model.predict(X_test)