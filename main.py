import load_dataset
import numpy as np
xtrain, xtest, ytrain, ytest = load_dataset.dataset()

print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)

rows, columns = xtrain.shape
x_matrix_train = np.zeros((rows, 6, 3, 1))
for i in range(rows):
    feature_matrix = xtrain.iloc[[i]].to_numpy()[0].reshape(6,3,1)
    x_matrix_train[i] = feature_matrix

print(x_matrix_train.shape)