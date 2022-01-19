#Importing libararies

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout,BatchNormalization
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn import metrics
from keras.models import Sequential, load_model




def make_model():
    
#Making a LSTM model
    
    model = Sequential()
    model.add(LSTM(128, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(trainY.shape[1]))

    model.compile(optimizer='adam', loss='mse')
    
    return model



def make_train_test_data():
#Reading test and train data
    

    with open('X_test_95_lstms.npy', 'rb') as f:
        X_test = np.load(f,allow_pickle=True)

    with open('X_train_95_lstms.npy', 'rb') as f:
        X_train = np.load(f,allow_pickle=True)

    with open('Y_test_95_lstms.npy', 'rb') as f:
        Y_test = np.load(f,allow_pickle=True)

    with open('Y_train_95_lstms.npy', 'rb') as f:
        Y_train = np.load(f,allow_pickle=True)


    #standardising the data 
    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    n_future = 7  # Number of days we want to predict for
    n_past =10    # Number of previous days we are looking to make prediction

     
    #Converting data into 3d matrix. where every entry has data for last 10 days as input and prices for upcoming 7 days as output    
        
    trainX = []
    trainY = []

    for i in range(n_past, len(X_train_scaled) - n_future +1):
        trainX.append(X_train_scaled[i - n_past:i])
        trainY.append(Y_train[i :i + n_future])


    trainX = np.array(trainX)
    trainY = np.array(trainY)


    testX = []
    testY = []

    for i in range(n_past, len(X_test_scaled) - n_future +1):
        testX.append(X_test_scaled[i - n_past:i])
        testY.append(Y_test[i :i + n_future])


    testX = np.array(testX)
    testY = np.array(testY)
    
    return trainX, trainY, testX, testY



def load_model_():
    
#Loading model
    
    model_ = load_model(filepath)
    return model_


def print_results(trainX, trainY, testX, testY):
    
    # making predictions
    y_pred = model.predict(testX)
    y_pred_train = model.predict(trainX)

    
    print('NEXT DAY PREDICTION')
    print('*'*100)
    print("TRAIN ERROR")

    print('MAE:', metrics.mean_absolute_error(trainY[0], y_pred_train[0]))  
    print('MSE:', metrics.mean_squared_error(trainY[0], y_pred_train[0]))  
    print('RMSE:', np.sqrt(metrics.mean_squared_error(trainY[0], y_pred_train[0])))
    print('*'*100)
    print("TEST ERROR")

    print('MAE:', metrics.mean_absolute_error(testY[0], y_pred[0]))  
    print('MSE:', metrics.mean_squared_error(testY[0], y_pred[0]))  
    print('RMSE:', np.sqrt(metrics.mean_squared_error(testY[0], y_pred[0])))
    
    print('NEXT 3 DAY PREDICTION')
    print('*'*100)
    print("TRAIN ERROR")

    print('MAE:', metrics.mean_absolute_error(trainY[0:3], y_pred_train[0:3]))  
    print('MSE:', metrics.mean_squared_error(trainY[0:3], y_pred_train[0:3]))  
    print('RMSE:', np.sqrt(metrics.mean_squared_error(trainY[0:3], y_pred_train[0:3])))
    print('*'*100)
    print("TEST ERROR")

    print('MAE:', metrics.mean_absolute_error(testY[0:3], y_pred[0:3]))  
    print('MSE:', metrics.mean_squared_error(testY[0:3], y_pred[0:3]))  
    print('RMSE:', np.sqrt(metrics.mean_squared_error(testY[0:3], y_pred[0:3])))




    print('NEXT 7 DAY PREDICTION')
    print('*'*100)
    print("TRAIN ERROR")

    print('MAE:', metrics.mean_absolute_error(trainY, y_pred_train))  
    print('MSE:', metrics.mean_squared_error(trainY, y_pred_train))  
    print('RMSE:', np.sqrt(metrics.mean_squared_error(trainY, y_pred_train)))
    print('*'*100)
    print("TEST ERROR")

    print('MAE:', metrics.mean_absolute_error(testY, y_pred))  
    print('MSE:', metrics.mean_squared_error(testY, y_pred))  
    print('RMSE:', np.sqrt(metrics.mean_squared_error(testY, y_pred)))
    
    
    
    print('Sample Prediction:')
    j=1770
    for i in range(n_future):
        print('Actual: ', testY[j][i], 'Predicted: ',y_pred[j][i])


print('Select one:')
print('1. Use trained model')
print('2. Train model')
print('Enter 1 or 2')
choice = int(input())
filepath = 'LSTM_7_model.hdf5'

n_future = 7  # Number of days we want to look into the future based on the past days.
n_past =10
trainX, trainY, testX, testY = make_train_test_data()

if choice == 1:
    
    model = load_model_()
    
else: 
    from keras.callbacks import ModelCheckpoint
    filepath = 'LSTM_7_model.hdf5'
    checkpoint = ModelCheckpoint(filepath=filepath, 
                                 monitor='val_loss',
                                 verbose=1, 
                                 save_best_only=True,
                                 mode='min')
    callbacks = [checkpoint]
    model = make_model()
    history = model.fit(trainX, trainY, epochs=25, batch_size=100, validation_split=0.1, verbose=2,callbacks=callbacks)

    
print_results(trainX, trainY, testX, testY)
    
