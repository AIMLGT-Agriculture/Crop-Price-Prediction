#Importing libararies

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import sklearn.model_selection as xval
from sklearn.datasets import fetch_openml
import forestci as fci
from sklearn import metrics
from sklearn.metrics import r2_score
import statistics
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout,BatchNormalization
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn import metrics
from sklearn import preprocessing



def months_cyclic_embedding(months):
    month_embedding = np.zeros((len(months),2))
    
    for i in range(len(months)):
        month_embedding[i][0] = np.sin(2 * np.pi * months[i]/11.0)
        month_embedding[i][1] = np.cos(2 * np.pi * months[i]/11.0)
        
    return month_embedding

def days_cyclic_embedding(days):
    day_embedding = np.zeros((len(days),2))
    
    for i in range(len(days)):
        day_embedding[i][0] = np.sin(2 * np.pi * days[i]/30.0)
        day_embedding[i][1] = np.cos(2 * np.pi * days[i]/30.0)
        
    return day_embedding


def make_price_nan_arrival_nan_array():

#Function to make a vectors for PriceNanFlag_array and ArrivalNanFlag_array. 
#These vectors will have true if previous day price was available otherwise false.

    PriceNanFlag_array = np.array(df['PriceNanFlag'])


    for i in range(len(PriceNanFlag_array)):
        if PriceNanFlag_array[i] ==True:
            PriceNanFlag_array[i]=1
        else:
            PriceNanFlag_array[i]=0

    PriceNanFlag_array = list(PriceNanFlag_array)

    PriceNanFlag_array = [0]+PriceNanFlag_array[:len(PriceNanFlag_array)-1]   #Appending [0] (False) as for 1st entry we do not have last day price
    
    ArrivalNanFlag_array = np.array(df['ArrivalNanFlag'])


    for i in range(len(ArrivalNanFlag_array)):
        if ArrivalNanFlag_array[i] ==True:
            ArrivalNanFlag_array[i]=1
        else:
            ArrivalNanFlag_array[i]=0

    ArrivalNanFlag_array = list(ArrivalNanFlag_array)

    ArrivalNanFlag_array = [0]+ArrivalNanFlag_array[:len(ArrivalNanFlag_array)-1]
    
    PriceNanFlag_array = np.array(PriceNanFlag_array)
    ArrivalNanFlag_array = np.array(ArrivalNanFlag_array)


    PriceNanFlag_array = PriceNanFlag_array.reshape([len(PriceNanFlag_array),1])
    ArrivalNanFlag_array = ArrivalNanFlag_array.reshape([len(ArrivalNanFlag_array),1])
    
    return PriceNanFlag_array,ArrivalNanFlag_array



def make_final_embedded_input(features,month_flag):
    
# Function to make final embedded input by concatenating different vectors to give final input to model

    month_array = np.array(month_input)
    day_array =np.array(day_input)

    PriceNanFlag_array,ArrivalNanFlag_array = make_price_nan_arrival_nan_array()
    
    
    input_ = df.values[:,features]
    if month_flag:
        Final_input = np.concatenate((input_,month_array),axis=1)
    else:
        Final_input = input_
    
    
    return Final_input


def time_based_split():
    
# Function to split data in train and test data. Till 2016 -> Train data and 2016-2018 -> Test data
    
    X_train =[]
    X_test=[]
    Y_train=[]
    Y_test=[]
    X_val=[]
    Y_val=[]
    years_df = list(df['Year'])
    
    for i in range(len(years_df)):
        if years_df[i] ==2018 or years_df[i]==2017:
            X_test.append(X[i])
            Y_test.append(Y[i])
        elif years_df[i] == 2016:
            X_val.append(X[i])
            Y_val.append(Y[i])
        else:
            X_train.append(X[i])
            Y_train.append(Y[i])
#         elif years_df[i] == 2016 or years_df[i] == 2015 or years_df[i] == 2014 or years_df[i] == 2013 or years_df[i] == 2012:

            
    return X_train,Y_train,X_test,Y_test,X_val,Y_val



def remove_outliers(X_train,Y_train,X_test,Y_test,X_val,Y_val,up,low):
    
# Function to remove outliers. We have used quantiles for removing outliers. For eg- if low if 5, then least 5% values will be removed
    X_train_95 = X_train.copy()
    X_test_95 = X_test.copy()
    X_val_95 = X_val.copy()
    Y_train_95 = Y_train.copy()
    Y_test_95 = Y_test.copy()
    Y_val_95 = Y_val.copy()

    high =np.percentile(Y, up)
    low = np.percentile(Y,low)

    outlier_indexes=[]


    for i in range(len(X_train)):
        if Y_train[i]<=low or Y_train[i]>=high:
            outlier_indexes.append(i)


    for index in sorted(outlier_indexes, reverse=True):
        del X_train_95[index]
        del Y_train_95[index]


    outlier_indexes=[]

    for i in range(len(X_test)):
        if Y_test[i]<=low or Y_test[i]>=high:
            outlier_indexes.append(i)


    for index in sorted(outlier_indexes, reverse=True):
        del X_test_95[index]
        del Y_test_95[index]
        
        
    outlier_indexes=[]

    for i in range(len(X_val)):
        if Y_val[i]<=low or Y_val[i]>=high:
            outlier_indexes.append(i)


    for index in sorted(outlier_indexes, reverse=True):
        del X_val_95[index]
        del Y_val_95[index]
        
        
        
    return X_train_95,X_test_95,Y_train_95,Y_test_95,X_val_95,Y_val_95


def market_OHV():
    markets = df['Market']
    market_names = list(df['Market'].unique())

    market_embedding_OHV = np.zeros((df.shape[0],len(market_names)))

    for i in range(len(markets)):
        market = markets[i]
        j = market_names.index(market)
        market_embedding_OHV[i][j]=1
        
    return market_embedding_OHV


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


def normalize_and_make_lstm_input(X_train,Y_train,X_test,Y_test,X_val,Y_val):
    n_future = 15  # Number of days we want to look into the future based on the past days.
    n_past =15

    scalerX = StandardScaler()
    scalerX = scalerX.fit(X_train)
    X_train_scaled = scalerX.transform(X_train)
    X_test_scaled = scalerX.transform(X_test)
    X_val_scaled = scalerX.transform(X_val)

    scalerY = StandardScaler()
    scalerY = scalerX.fit(np.array(Y_train).reshape(-1,1))
    Y_train_scaled = scalerY.transform(np.array(Y_train).reshape(-1,1))
    Y_test_scaled = scalerY.transform(np.array(Y_test).reshape(-1,1))
    Y_val_scaled = scalerY.transform(np.array(Y_val).reshape(-1,1))


    trainX = []
    trainY = []

    for i in range(n_past, len(X_train_scaled) - n_future +1):
        trainX.append(X_train_scaled[i - n_past:i])
        trainY.append(Y_train_scaled[i :i + n_future])

    trainX = np.array(trainX)
    trainY = np.array(trainY)

    testX = []
    testY = []

    for i in range(n_past, len(X_test_scaled) - n_future +1):
        testX.append(X_test_scaled[i - n_past:i])
        testY.append(Y_test_scaled[i :i + n_future])

    testX = np.array(testX)
    testY = np.array(testY)

    valX = []
    valY = []

    for i in range(n_past, len(X_val_scaled) - n_future +1):
        valX.append(X_val_scaled[i - n_past:i])
        valY.append(Y_val_scaled[i :i + n_future])

    valX = np.array(valX)
    valY = np.array(valY)
    
    return trainX,testX,valX,trainY,testY,valY,scalerY

def print_rmse(actual,predicted,scalerY):
    
    a=scalerY.inverse_transform(actual)
    a=a.reshape((a.shape[0],a.shape[1]))
    b=scalerY.inverse_transform(predicted)
    return np.sqrt(metrics.mean_squared_error(a,b))
    
    




#reading csv file into dataframe
df = pd.read_csv('df_with_updated_lat_long_and_last_year_price_2009_2018.csv')
df = df.drop(['Unnamed: 0'],axis=1)

#One hot vector embeddings for market
market_input= market_OHV()

#month embedding
months_array= df['Month']
month_input = months_cyclic_embedding(months_array)

# day embedding    
days_array= df['Day']
day_input = days_cyclic_embedding(days_array)


#Making final input by concatenating the embeddings
Final_input = make_final_embedded_input([21,24,33,29,32,25,9,10],True)



# Y- price (output)
Y= df.values[:,5]
X = Final_input

X_train,Y_train,X_test,Y_test,X_val,Y_val = time_based_split()
X_train,X_test,Y_train,Y_test,X_val,Y_val= remove_outliers(X_train,Y_train,X_test,Y_test,X_val,Y_val,95,5)

trainX,testX,valX,trainY,testY,valY,scalerY =normalize_and_make_lstm_input(X_train,Y_train,X_test,Y_test,X_val,Y_val)



model = make_model()

history = model.fit(trainX, trainY, epochs=35, batch_size=256, validation_data=(valX,valY), verbose=2)

y_pred = model.predict(testX)
y_pred_train = model.predict(trainX)

print('RMSE on train: ',print_rmse(trainY,y_pred_train,scalerY))
print('RMSE on test: ',print_rmse(testY,y_pred,scalerY))
