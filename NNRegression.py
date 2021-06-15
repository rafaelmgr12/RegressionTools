## Import libs
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.insert(1, 'modules/')

import data_processing as dp
import wrangle
import keras
import talos
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation

from tensorflow.keras.optimizers import Adam
from keras import backend as K
import keras as ks
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import layers
from tensorflow.keras import regularizers

from sklearn.preprocessing import StandardScaler
std = StandardScaler()
std1 = StandardScaler()

data = pd.read_csv('datasets/USA_Housing.csv')
X = data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']].values
X = std.fit_transform(X)

y = data['Price'].values
y = std1.fit_transform(y.reshape(-1,1))
y = dp.discritizer_target(y)

x_train, y1_train, x_val, y1_val = wrangle.array_split(X, y[:,:100], 0.3) # 100 is the number of bins, if you choose another change it!
x_train, y2_train, x_val, y2_val = wrangle.array_split(X, y[:,100], 0.3)

def telco_churn(x_train, y_train, x_val, y_val, params, n_bins = 100):

    # the second side of the network
    input_layer = keras.layers.Input(shape=(5,))
    hidden_layer1 = Dense(params['first_layer'],
                          activation=params['activation'])(input_layer)
    hidden_layer2 = Dense(params['second_layer'],
                          activation=params['activation'])(hidden_layer1)
    hidden_layer3 = Dense(params['third_layer'],
                          activation=params['activation'])(hidden_layer2)

    # creating the outputs
    output1 = Dense(n_bins,  activation='softmax', name='pdf')(hidden_layer3)
    output2 = Dense(1, activation='linear', name='reg')(hidden_layer3)

    losses = {"pdf": keras.losses.CategoricalCrossentropy(),
              "reg": "mean_absolute_error"}

    loss_weights = {"pdf": 0.9, "reg": 0.1}

    # put the model together, compile and fit
    model = keras.Model(inputs=input_layer, outputs=[output1, output2])

    model.compile(params["compile"], loss=losses, loss_weights=loss_weights,
                  metrics={'pdf': "acc",
                           'reg': "mse"})

    out = model.fit(x=x_train,
                    y=y_train,
                    #validation_data=[x_val, y_val],
                    validation_split=0.2,
                    epochs=150,
                    batch_size=params['batch_size'],
                    verbose=0)

    return out, model
p = {'activation':['relu',"tanh"],
     'first_layer': [10,20,30],
     'second_layer': [10,20],
     'third_layer': [10],
     'batch_size': [32,64],
     'compile' : ["adam",keras.optimizers.RMSprop()]}

scan_object = talos.Scan(x=x_train,
                         y={"pdf":y1_train, "reg":y2_train},
                         x_val=x_val,
                         y_val=[y1_val, y2_val],
                         params=p,
                         model=telco_churn, experiment_name="name")
model = scan_object.best_model(metric='reg_mse', asc=False)
stats = scan_object.data
stats.to_csv('statistics.csv',index = False)