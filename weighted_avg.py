from rnn_1 import Rnn
from cnn import Cnn
from dt import Dt
from rnn import Ann
import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_curve
from matplotlib import pyplot
dataset = load_dataset.dataset()
obj_rnn = Rnn(load_dataset.matrix_dataset(p_dataset=dataset))
rnn_model = obj_rnn.model

obj_cnn = Cnn(load_dataset.matrix_dataset(p_dataset=dataset))
cnn_model = obj_cnn.model

obj_dt = Dt(dataset)
dt_model = obj_dt.model

obj_ann = Ann(dataset)
ann_model = obj_ann.model

threshold = 0.5
prediction = np.empty(obj_dt.prediction.shape)
weights = (0.31, 0.23, 0.23, 0.23)
predictions = (obj_cnn.prediction, obj_dt.prediction, obj_rnn.prediction, obj_ann.prediction)
for i in range(obj_dt.prediction.shape[0]):
    avg = (predictions[0][i] * weights[0] + predictions[1][i] * weights[1] + predictions[2][i] * weights[2])

    val = 1 if avg > threshold else 0
    prediction[i] = val

X_train, X_test, y_train, y_test = dataset
y_test = np.argmax(y_test, axis=1)
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

