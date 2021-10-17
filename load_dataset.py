import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
Matrix_x, Matrix_y = (4,4)
features_file = "features.txt"
run_no = "new_output_folder"


max_val = 99999

with open(features_file) as f:
    features = [feature.strip() for feature in f]


with open('train_files.txt') as f:
    train_files = [filename.strip() for filename in f]

with open('test_files.txt') as f:
    test_files = [filename.strip() for filename in f]

train_dir = 'CSV-01-12/01-12/'

test_dir = 'CSV-03-11/03-11/'
output_dir = '../cedar/output/'


def read_file(filename, y_out):
    df = pd.read_csv(filename, nrows=1000) # number of rows to read from each dataset's file
    df.columns = df.columns.str.strip()
    df = df[features]
    NewLabel = []
    for i in df["Label"]:
        if i =="BENIGN":
            NewLabel.append(0)
        else:
            NewLabel.append(1)
    df["Label"]=NewLabel
    y = df['Label'].values
    y_out = y_out.extend(y)
    del df['Label']

    df = df.replace('Infinity', max_val)
    x = df.values
    scaler = QuantileTransformer(n_quantiles=1000, random_state=42)
    scaled_df = scaler.fit_transform(x)

    x = pd.DataFrame(scaled_df)
    return x

def dataset():
    nClasses = 2


    new_x = pd.DataFrame()
    temp_y = []

    for f in train_files:
        print('Processing file ' + f + '\n')
        new_x = new_x.append(read_file(train_dir + f, temp_y))
        print('Processed file ' + f + ' , total samples is ' + str(len(temp_y)) + '\n')

    new_y = to_categorical(temp_y, num_classes=nClasses)

    xTrain, xTest, yTrain, yTest = train_test_split(new_x, new_y, test_size=0.2, random_state=42)

    print('train size: ', xTrain.shape)
    print('train labels: ', yTrain.shape)
    return xTrain, xTest, yTrain, yTest


def convert_vector_to_image_matrix(x):
    rows, columns = x.shape
    x_matrix_train = np.zeros((rows, Matrix_x, Matrix_y))
    for i in range(rows):
        feature_matrix = x.iloc[[i]].to_numpy()[0].reshape(Matrix_x, Matrix_y)
        x_matrix_train[i] = feature_matrix

    return x_matrix_train


def matrix_dataset(p_dataset=None):
    if p_dataset is None:
        xtrain, xtest, ytrain, ytest = dataset()
    else:
        xtrain, xtest, ytrain, ytest = p_dataset
    print(xtrain.shape)
    print(xtest.shape)
    print(ytrain.shape)
    print(ytest.shape)

    xtrain = convert_vector_to_image_matrix(xtrain)
    xtest = convert_vector_to_image_matrix(xtest)
    return xtrain, xtest, ytrain, ytest