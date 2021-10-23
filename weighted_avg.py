from ROC import roc
from rnn_1 import Rnn
from cnn import Cnn
from dt import Dt
from ann import Ann
import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_curve

x_train, y_train = load_dataset.dataset()
x_test, y_test = load_dataset.next_day_data()
dataset = (x_train, x_test, y_train, y_test)

obj_rnn = Rnn(load_dataset.matrix_dataset(p_dataset=dataset))
rnn_model = obj_rnn.model

obj_cnn = Cnn(load_dataset.matrix_dataset(p_dataset=dataset))
cnn_model = obj_cnn.model

obj_dt = Dt(dataset)
dt_model = obj_dt.model

obj_ann = Ann(dataset)
ann_model = obj_ann.model

avg_probs = []
weights = (0.31, 0.23, 0.23, 0.23)
prob = (obj_cnn.prob, obj_dt.prob, obj_rnn.prob, obj_ann.prob)
for i in range(obj_dt.prediction.shape[0]):

    class0 = (prob[0][i][0] * weights[0] + prob[1][i][0] * weights[1] + prob[2][i][0] * weights[2])
    class1 = (prob[0][i][1] * weights[0] + prob[1][i][1] * weights[1] + prob[2][i][1] * weights[2])
    avg = [class0/(class0 + class1), class1/(class0 + class1)]
    avg_probs.append(avg)

avg_probs = np.array(avg_probs)
score = avg_probs[:, 1]

prediction = np.argmax(avg_probs, axis=1)
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

roc(score, y_test)