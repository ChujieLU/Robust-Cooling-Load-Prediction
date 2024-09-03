
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from scipy import interpolate
np.random.seed(7)
import tensorflow as tf
from tensorflow.keras.layers import *


data = pd.read_csv("goffice.csv") # import data, here taking the office building as an example

train_raw=data[0:3480][['WD','Solar','To','hour','weekday','CL']]
test_raw=data[3480:][['WD','Solar','To','hour','weekday','CL']]

min_max_scaler = preprocessing.MinMaxScaler()
train_nor=min_max_scaler.fit_transform(train_raw)
test_nor=min_max_scaler.transform(test_raw)

#Preprosing: Sliding window
def create_dataset(data,n_predictions,n_next):

    dim = data.shape[1]
    train_X, train_Y = [], []
    for i in range(data.shape[0]-n_predictions-n_next-1):
        a = data[i:(i+n_predictions),:]
        train_X.append(a)
        tempb = data[i+n_predictions+n_next,5]
        train_Y.append(tempb)
    train_X = np.array(train_X,dtype='float64')
    train_Y = np.array(train_Y,dtype='float64')

    return train_X, train_Y

X_width=24
y_width=1

trainX,trainy = create_dataset(train_nor,X_width,y_width)
testX,testy = create_dataset(test_nor,X_width,y_width)

#Define T2V Layer
class T2V(Layer):

    def __init__(self, output_dim=None, **kwargs):
        self.output_dim = output_dim
        super(T2V, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[-1], self.output_dim),
                                 initializer='uniform',
                                 trainable=True)

        self.P = self.add_weight(name='P',
                                 shape=(input_shape[1], self.output_dim),
                                 initializer='uniform',
                                 trainable=True)

        self.w = self.add_weight(name='w',
                                 shape=(input_shape[1], 1),
                                 initializer='uniform',
                                 trainable=True)

        self.p = self.add_weight(name='p',
                                 shape=(input_shape[1], 1),
                                 initializer='uniform',
                                 trainable=True)

        super(T2V, self).build(input_shape)

    def call(self, x):
        original = self.w * x + self.p
        sin_trans = K.sin(K.dot(x, self.W) + self.P)

        return K.concatenate([sin_trans, original], -1)

#   Model construction
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Conv1D, MaxPooling1D,Dropout,concatenate,Permute,Reshape,multiply
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense,Bidirectional,Input
from tensorflow.keras.layers import RepeatVector, TimeDistributed
from sklearn.metrics import r2_score,mean_squared_error
from keras import callbacks
from keras import backend as K

SINGLE_ATTENTION_VECTOR = False

def attention_3d_block(inputs):
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs])
    return output_attention_mul


def TAttLSTM1():
    inputs= Input(shape=(TIME_STEPS, INPUT_DIM,))
    t2v=T2V(32)(inputs)
    lstm=LSTM(16,return_sequences=1)(t2v)
    att=attention_3d_block(lstm)
    att_out=Flatten()(att)
    drop=Dropout(0.1)(att_out)
    dense=Dense(50,activation='relu')(drop)
    output=Dense(1,activation='linear')(dense)
    model=Model(inputs,output)
    return model

INPUT_DIM = trainX.shape[2]
TIME_STEPS = trainX.shape[1]

def inverse_transform_col(scaler,y,n_col):
    y = y.copy()
    y -= scaler.min_[n_col]
    y /= scaler.scale_[n_col]
    return y


lastR2 = []
lastRMSE = []
lastCVRMSE = []

for i in range(5):
    print('iteration: ', i)

    model = TAttLSTM1()
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    batch_size = 64
    epoch_num = 200
    history = model.fit(trainX, trainy, batch_size=batch_size, epochs=epoch_num, verbose=0)
    print('\n', model.summary())

    yPreds_nor = model.predict(testX)
    testy_raw = inverse_transform_col(min_max_scaler, testy, n_col=5)
    yPreds = inverse_transform_col(min_max_scaler, yPreds_nor, n_col=5)

    r2 = r2_score(testy_raw, yPreds)
    rmse = np.sqrt(mean_squared_error(testy_raw, yPreds))
    m = testy_raw.mean()
    cvrmse = rmse / m
    print("R2:", r2)
    print("RMSE:", rmse)
    print("RMSE:", cvrmse)

    lastR2.append(r2)
    lastRMSE.append(rmse)
    lastCVRMSE.append(cvrmse)

    K.clear_session()

print(lastR2)
print(lastRMSE)
print(lastCVRMSE)

lastR2=pd.DataFrame(lastR2)
lastRMSE=pd.DataFrame(lastRMSE)
lastCVRMSE=pd.DataFrame(lastCVRMSE)
print(lastR2.mean(),lastR2.std())
print(lastRMSE.mean(),lastRMSE.std())
print(lastCVRMSE.mean(),lastCVRMSE.std())