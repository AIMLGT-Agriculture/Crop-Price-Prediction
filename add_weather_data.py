
#importing required liberaries
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
import math
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler


#Reading imputed data into dataframe
df = pd.read_csv('Wheat/Wheat/Wheat__UPDATED_all_mandis_post_impute.csv')


#Sorting dataframe
df = df.sort_values(by = ['District','Market', 'Year','Month','Day'])
df = df.reset_index(drop=True)

#Dropping extra column
df= df.drop(['Unnamed: 0'], axis=1)



#reading latitude and longitude of districts from pkl files
import pickle


with open('lat_district_dict.pkl', 'rb') as f:
    dict_lat = pickle.load(f)
    
with open('long_district_dict.pkl', 'rb') as f:
    dict_long = pickle.load(f)
    



long_list=[]
lat_list=[]


#Adding lat long for every entry of dataframe
for i in range(df.shape[0]):
    district = df['District'][i]
    long_list.append(dict_long[district])
    lat_list.append(dict_lat[district])
    

df['latitude']=lat_list
df['longitude']=long_list




# Adding weather data

#Reading environmental data
df_env = pd.read_csv('Environmental_data_date_wise.csv')
df_env= df_env.drop(['Unnamed: 0'], axis=1)


month_dict = {'01':'Jan','02':'Feb', '03':'Mar','04':'Apr','05': 'May','06': 'Jun','07':'Jul','08': 'Aug','09':'Sep','10': 'Oct','11':'Nov','12':'Dec'}

dates=[]
month_names=[]
years=[]
month=[]

#Adding month, month name, year, day columns to environmental dataframe
for i in range(df_env.shape[0]):
  dates.append(int(df_env['time'][i][8:10]))
  month.append(int(df_env['time'][i][5:7]))
  month_names.append(month_dict[(df_env['time'][i][5:7])])
  years.append(df_env['time'][i][:4])
    
dates__ = []
for i in range(len(month)):
    date = (df_env['time'][i][8:10])+'-'+month_names[i]+'-'+years[i]
    dates__.append(date)
df_env['Day'] = dates
df_env['Month_name'] = month_names
df_env['Year'] = years
df_env['Month'] = month
df_env['Date'] = dates__

df_env = df_env.drop(['time'],axis=1)

#Unique lat long in environmental data
target_lat= df_env['latitude'].unique()
target_long=df_env['longitude'].unique()

#Unique lat long in imputed price arrival data
lat=df['latitude'].unique()
long=df['longitude'].unique()


# Making dictionaries for lat long to map the lat long of price arrival data to available
#lat long in environmental data

# for eg if in price data lat is 28.75 and in environmental data nearest lat is 29 then we will make 
# 28.75 to 29 in lat dict so that we can later add environmental data for lat 29 to price data having lat 28.75
lat_dict={}
for l in lat:
    lat_dict[l]=0



for l in lat:
    cur =100
    for t in target_lat:
        if abs(l-t)<cur:
            cur = abs(l-t)
            lat_dict[l]=t
            
long_dict={}
for l in long:
    long_dict[l]=0



for l in long:
    cur =100
    for t in target_long:
        if abs(l-t)<cur:
            cur = abs(l-t)
            long_dict[l]=t

updated_long=[]
updated_lat=[]

for i in range(df.shape[0]):
    lat = df['latitude'][i]
    long = df['longitude'][i]
    
    updated_long.append(long_dict[long])
    updated_lat.append(lat_dict[lat])
    
df['latitude_env'] = updated_lat
df['longitude_env'] = updated_long

df.head()

lat=df['latitude_env']
long = df['longitude_env']

df['latitude'] = lat
df['longitude'] = long



#Merging price arrival data with environmental data

df= df.drop(['latitude_env','longitude_env'], axis=1)
df_new = pd.merge(df, df_env, on=['Date','latitude','longitude'])
df_new.head()

df_new= df_new.drop(['Day_y', 'Month_name_y', 'Year_y', 'Month_y'], axis=1)

df_new['Month'] = df_new['Month_x']
df_new['Month_name'] = df_new['Month_name_x']
df_new['Day'] = df_new['Day_x']
df_new['Year'] = df_new['Year_x']

df_new= df_new.drop(['Month_name_x', 'Day_x', 'Year_x', 'Month_x'], axis=1)



# sorting datafra,e
df_new = df_new.sort_values(by = ['District','Market', 'Year','Month','Day'])
df_new = df_new.reset_index(drop=True)


# Saving to a CSV file
df_new.to_csv('Wheat/Wheat/Wheat__UPDATED_all_mandis_env_post_impute.csv')

