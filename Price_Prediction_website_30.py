#!/usr/bin/env python
# coding: utf-8

# In[61]:


import pandas as pd
import numpy as np
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





def make_price_nan_arrival_nan_array():
    PriceNanFlag_array = np.array(df['PriceNanFlag'])


    for i in range(len(PriceNanFlag_array)):
        if PriceNanFlag_array[i] ==True:
            PriceNanFlag_array[i]=1
        else:
            PriceNanFlag_array[i]=0

    PriceNanFlag_array = list(PriceNanFlag_array)

    PriceNanFlag_array = [0]+PriceNanFlag_array[:len(PriceNanFlag_array)-1]
    
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



def make_final_embedded_input():
    market_array = np.array(market_input)
    day_array =np.array(day_input)
    month_array = np.array(month_input)
    PriceNanFlag_array,ArrivalNanFlag_array = make_price_nan_arrival_nan_array()
    
    
    input_ = df.values[:,[9,10,11,12,13,14,15,16, 21,22,23,24,25,26,27,28,29,30,31,32]]
    Final_input = np.concatenate((input_,PriceNanFlag_array,ArrivalNanFlag_array,market_array,month_array,day_array),axis=1)
    
    
    return Final_input


def time_based_split():
    X_train =[]
    X_test=[]
    Y_train=[]
    Y_test=[]
    years_df = list(df['Year'])
    
    for i in range(len(years_df)):
        if years_df[i] ==2018 or years_df[i]==2017:
            X_test.append(X[i])
            Y_test.append(Y[i])
        else:
            X_train.append(X[i])
            Y_train.append(Y[i])
            
    return X_train,Y_train,X_test,Y_test



def remove_outliers(X_train,Y_train,X_test,Y_test,up,low):
    X_train_95 = X_train.copy()
    X_test_95 = X_test.copy()
    Y_train_95 = Y_train.copy()
    Y_test_95 = Y_test.copy()

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
        
    return X_train_95,X_test_95,Y_train_95,Y_test_95



def fit_rf(X_train,Y_train,X_test,Y_test, n_estimators=100,max_depth=7,verbose=2):
 
    regressor = RandomForestRegressor(n_estimators = 100,max_depth=7,verbose=2, random_state = 0)

    # fit the regressor with x and y data
    regressor.fit(X_train, Y_train)

    y_pred = regressor.predict(X_test) 
    y_pred_train = regressor.predict(X_train)
    
    print('*'*100)
    print("TRAIN ERROR")

    print('MAE:', metrics.mean_absolute_error(Y_train, y_pred_train))  
    print('MSE:', metrics.mean_squared_error(Y_train, y_pred_train))  
    print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_train, y_pred_train)))
    
    print('*'*100)
    print("TEST ERROR")
    
    print('MAE:', metrics.mean_absolute_error(Y_test, y_pred))  
    print('MSE:', metrics.mean_squared_error(Y_test, y_pred))  
    print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))
    
    print('*'*100)
    for i in range(10):
        print('Pred: ',y_pred[i],' True: ',Y_test[i])
        
    return regressor



from sklearn import linear_model
def fit_lasso(X_train,Y_train,X_test,Y_test):
 
    rgsrL = linear_model.Lasso(alpha=0.1)
    rgsrL.fit(X_train,Y_train)

    y_pred= rgsrL.predict(X_test)

    y_pred_train = rgsrL.predict(X_train)
    print('*'*100)
    print("TRAIN ERROR")

    print('MAE:', metrics.mean_absolute_error(Y_train, y_pred_train))  
    print('MSE:', metrics.mean_squared_error(Y_train, y_pred_train))  
    print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_train, y_pred_train)))
    
    print('*'*100)
    print("TEST ERROR")
    
    print('MAE:', metrics.mean_absolute_error(Y_test, y_pred))  
    print('MSE:', metrics.mean_squared_error(Y_test, y_pred))  
    print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))
    
    print('*'*100)
    for i in range(10):
        print('Pred: ',y_pred[i],' True: ',Y_test[i])


#Reading dataframe
df = pd.read_csv('updated_impute_features_added_post_impute_environmental_43_markets_2008_2018.csv')
df = df.drop(['Unnamed: 0'],axis=1)

# Embedded inputs
df_market_embedding = pd.read_csv('market_embedding.csv')
df_market_embedding = df_market_embedding.drop(['Unnamed: 0'],axis=1)
df_market_embedding = df_market_embedding.reset_index(drop=True)

markets = df['Market'].unique()
markets = list(markets)

market_array= df['Market']

market_input=[]
for i in range(df.shape[0]):
    market = df['Market'][i]
    index = markets.index(market)
    market_input.append([df_market_embedding['C1'][index],df_market_embedding['C2'][index],df_market_embedding['C3'][index]])
    
    
df_month_embedding = pd.read_csv('month_embedding.csv')
df_month_embedding = df_month_embedding.drop(['Unnamed: 0'],axis=1)
df_month_embedding = df_month_embedding.reset_index(drop=True)

months = df['Month'].unique()
months = list(months)

month_array= df['Month']


month_input=[]
for i in range(df.shape[0]):
    month = df['Month'][i]
    index = months.index(month)
    month_input.append([df_month_embedding['C1'][index],df_month_embedding['C2'][index],df_month_embedding['C3'][index]])
    
    
df_day_embedding = pd.read_csv('day_embedding.csv')
df_day_embedding = df_day_embedding.drop(['Unnamed: 0'],axis=1)
df_day_embedding = df_day_embedding.reset_index(drop=True)

days = df['Day'].unique()
days = list(days)

day_array= df['Day']


day_input=[]
for i in range(df.shape[0]):
    day = df['Day'][i]
    index = days.index(day)
    day_input.append([df_day_embedding['C1'][index],df_day_embedding['C2'][index],df_day_embedding['C3'][index]])
    
    

    
    
    
    
    
features = ['latitude',
 'longitude',
 'Soil_Type',
 'Humidity',
 'Sunlight',
 'Temperature',
 'Avg_Precipitation',
 'Last_Precipitation',
 'last_day_price',
 'second_last_day_price',
 'third_last_day_price',
 'last_week_avg_price',
 'last_day_arrival',
 'second_last_day_arrival',
 'third_last_day_arrival',
 'last_week_avg_arrival',
 'last_day_Precipitation',
 'second_last_day_Precipitation',
 'third_last_day_Precipitation',
 'last_week_avg_Precipitation',
 'PriceNanFlag',
 'ArrivalNanFlag',
 'Market',
 'Month',
 'Day']  

features_bool ={}
for f in features:
    features_bool[f] = True
    
# # ********************************************************************
# Need a scrolling list where user can select what features to eliminate
# **********************************************************************
    
columns = df.columns   
deleted_features = [1,2,4,9,10,11,12,13,14,15,17]  

updated_deleted_features =[]

for i in deleted_features:
    if i==24:
        updated_deleted_features = updated_deleted_features+[28,29,30]
    elif i==23:
        updated_deleted_features = updated_deleted_features+[25,26,27]
    elif i==22:
        updated_deleted_features = updated_deleted_features+[22,23,24]
    else:
        updated_deleted_features.append(i)


Final_input = make_final_embedded_input()


# XX = Final_input.copy()

for index in sorted(updated_deleted_features, reverse=True):
    Final_input = np.delete(Final_input,index,axis =1)

Y= df.values[:,5]

X = Final_input

X_train,Y_train,X_test,Y_test = time_based_split()

X_train_95,X_test_95,Y_train_95,Y_test_95 = remove_outliers(X_train,Y_train,X_test,Y_test,95,5)

rgsrr = fit_rf(X_train_95,Y_train_95,X_test_95,Y_test_95,100,10,0)


importances = rgsrr.feature_importances_
importances = np.multiply(importances,10000000)

indexes=[9,10,11,12,13,14,15,16, 21,22,23,24,25,26,27,28,29,30,31,32]
features = list(df.columns)
input_features =[]

for i in indexes:
    input_features.append(features[i])
    
# for i in range(len(input_features)):
#     print(indexes[i],' : ',input_features[i],' : ', importances[i])

    
input_features = input_features + ['PriceNanFlag','ArrivalNanFlag','Market','Month','Day']


for index in sorted(deleted_features, reverse=True):
    del input_features[index]
    
    
    



n=len(importances)

extra=[]

if 'Day' in input_features:
    day_importance = 0
    for i in range(n-3,n):
        day_importance +=importances[i]
    day_importance /=3
    n -=3
    extra.append(day_importance)

    
    
if 'Month' in input_features:
    month_importance = 0
    for i in range(n-3,n):
        month_importance +=importances[i]
    month_importance /=3
    n-=3
    extra.append(month_importance)
    
if 'Market' in input_features:
    market_importance = 0
    for i in range(n-3,n):
        market_importance +=importances[i]
    market_importance /=3
    n-=3
    extra.append(market_importance)



input_features_importance= list(importances[:n])
# input_features_importance = input_features_importance + [market_importance,month_importance ,day_importance]
input_features_importance = input_features_importance + extra

for i in range(len(input_features)):
    print(input_features[i],' : ', input_features_importance[i])
    
sum_=0
for i in input_features_importance:
    sum_ +=i
    
for i in range(len(input_features_importance)):
    input_features_importance[i] = input_features_importance[i]/sum_
    
    
import numpy as np
import matplotlib.pyplot as plt




fig = plt.figure(figsize =(10, 7))

# creating the bar plot
plt.bar(input_features,input_features_importance)
plt.xticks(rotation='vertical')
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.show()



if 'Month' in input_features and 'Day' in input_features and "Market" in input_features:
    market = 'Agra'
    day = 12
    month =5



    a=[]
    index = markets.index(market)
    a.append([df_market_embedding['C1'][index],df_market_embedding['C2'][index],df_market_embedding['C3'][index]])


    b=[]
    index = months.index(month)
    b.append([df_month_embedding['C1'][index],df_month_embedding['C2'][index],df_month_embedding['C3'][index]])

    c=[]
    index = days.index(day)
    c.append([df_day_embedding['C1'][index],df_day_embedding['C2'][index],df_day_embedding['C3'][index]])


    check = a[0]+b[0]+c[0]

    index =0
    for i in range(len(X_test_95)):
        ar = list(X_test_95[i][-9:])

        if ar == check:
    #         print('Yes')
            index =i



    y_pred_30 = rgsrr.predict(np.array(X_test[index:index+30]).reshape(30,-1))


    for i in range(30):
        print('Day :',i,' Actual Price: ',Y_test[index+i],' Predicted Price: ',y_pred_30[i])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[54]:


market = 'Agra'
day = 12
month =5



a=[]
index = markets.index(market)
a.append([df_market_embedding['C1'][index],df_market_embedding['C2'][index],df_market_embedding['C3'][index]])
    

b=[]
index = months.index(month)
b.append([df_month_embedding['C1'][index],df_month_embedding['C2'][index],df_month_embedding['C3'][index]])
    
c=[]
index = days.index(day)
c.append([df_day_embedding['C1'][index],df_day_embedding['C2'][index],df_day_embedding['C3'][index]])


check = a[0]+b[0]+c[0]

index =0
for i in range(len(X_test_95)):
    ar = list(X_test_95[i][-9:])
    
    if ar == check:
#         print('Yes')
        index =i

        
        
y_pred_30 = rgsrr.predict(np.array(X_test[index:index+30]).reshape(30,-1))


for i in range(30):
    print('Day :',i,' Actual Price: ',Y_test[index+i],' Predicted Price: ',y_pred_30[i])


# In[60]:


'Month' in input_features
    


# In[ ]:




