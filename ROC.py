import matplotlib.pyplot as plt
import numpy as np
#
# score = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.54, 0.53, 0.52, 0.51, 0.505, 0.4, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34, 0.33, 0.30, 0.1])
# y = np.array([1,1,0, 1, 1, 1, 0, 0, 1, 0, 1,0, 1, 0, 0, 0, 1 , 0, 1, 0])


def roc(score, y):
    # false positive rate
    fpr = []
    # true positive rate
    tpr = []
    # Iterate thresholds from 0.0, 0.01, ... 1.0
    thresholds = np.arange(0.0, 1.01, .01)

    # get number of positive and negative examples in the dataset
    P = sum(y)
    N = len(y) - P

    # iterate through all thresholds and determine fraction of true positives
    # and false positives found at this threshold
    for thresh in thresholds:
        FP=0
        TP=0
        for i in range(len(score)):
            if (score[i] > thresh):
                if y[i] == 1:
                    TP = TP + 1
                if y[i] == 0:
                    FP = FP + 1
        fpr.append(FP/float(N))
        tpr.append(TP/float(P))

    plt.plot(fpr, tpr)
    plt.show()