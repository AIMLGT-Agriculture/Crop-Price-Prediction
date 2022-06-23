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


#Reading raw price and arrival data into dataframe from csv format
mod_df = pd.read_csv('Wheat/Wheat/Wheat_UP_Arrival_Raw.csv')          #Arrival data
price_df = pd.read_csv('Wheat/Wheat/Wheat_UP_Price_Raw.csv')          # Price data


#Price data and arrival data has dates in different string format 
# in arrival data dates are in format dd-mm-yyyy, whereas in price data dates are in format dd mm yyyy
# i.e, hyphen(-) are replaced by spaces in price data. Therefore to make both dates similar we are doing following

for i in range(price_df.shape[0]):
    price_df['Price Date'][i] =price_df['Price Date'][i][:2]+'-'+price_df['Price Date'][i][3:6]+'-'+price_df['Price Date'][i][7:]
    
    
#Renaming Price Date column as Date in price data
price_df.rename(columns = {'Price Date':'Date'}, inplace = True)


#Dropping columns that are not required for merging and imputation
price_df= price_df.drop(['State','State Code','District Code','Market Code','Commodity','Commodity Code','Variety','Grade',
                        'Min Price','Max Price'], axis=1)

mod_df= mod_df.drop(['State','State Code','Commodity','Commodity Code'], axis=1)

#Renaming Volume Date column as Date in arrival data
mod_df.rename(columns = {'Volume Date':'Date'}, inplace = True)


#Merging price and arrival data
df_new = pd.merge(mod_df,price_df,on=['Date','District','Market'])
mod_df = df_new

#mandis in data
mandi_list = mod_df['Market'].unique()

#Sorting by markets and date
mod_df = mod_df.sort_values(by = ['Market','Date'])
mod_df = mod_df.reset_index(drop=True)

# Districts and markets in data
districts = mod_df['District'].unique()
markets = mod_df['Market'].unique()




# Adding month, month_name, day, year columns using date columns

df=mod_df

month_array = []
day_array=[]
year_array=[]
for i in range(df.shape[0]):
    month_array.append(df["Date"][i][3:6])
    day_array.append(int(df['Date'][i][0:2]))
    year_array.append(int(df['Date'][i][7:]))
    
df['Month_name']= month_array
df['Day'] = day_array
df['Year']= year_array

month_names = np.array(df['Month_name'])
month_names = list(month_names) 
months = month_names.copy()

for i in range(len(months)):
  if month_names[i] == "Jan":
    months[i] = 1
  elif month_names[i] == "Feb":
    months[i] = 2
  elif month_names[i] == "Mar":
    months[i] = 3
  elif month_names[i] == "Apr":
    months[i] = 4
  elif month_names[i] == "May":
    months[i] = 5
  elif month_names[i] == "Jun":
    months[i] = 6
  elif month_names[i] == "Jul":
    months[i] = 7
  elif month_names[i] == "Aug":
    months[i] = 8
  elif month_names[i] == "Sep":
    months[i] = 9
  elif month_names[i] == "Oct":
    months[i] = 10
  elif month_names[i] == "Nov":
    months[i] = 11
  elif month_names[i] == "Dec":
    months[i] = 12

df['Month']=months


#Removing date of 2008
mod_df = df.loc[df['Year'] != 2008]
mod_df = mod_df.reset_index(drop=True)


#Reading a file which contains all the dates from 2008-18. This file will be used to 
#recognise the missing dates in our data

df = pd.read_csv('df_with_last_year_price_2009_2018.csv')
df = df.drop(['Unnamed: 0'],axis=1)
df = df.loc[df['Year'] != 2008]

# All dates from 2009-18
dates = df['Date'].unique()

# Sorting data and dropping columns
mod_df = mod_df.sort_values(by = ['Market','Year','Month','Day'])
mod_df =mod_df.drop(['Month_name', 'Day', 'Year', 'Month'],axis=1)
mod_df = mod_df.reset_index(drop=True)



# We are adding dummy entries in new_df_list for the missing dates data

new_df_list=[]
count=0
for market in markets:
    
    #Considering market wise data
    cur_df = mod_df.loc[mod_df['Market'] == market]
    
    #Available date for the considered market data
    cur_dates = np.array(cur_df['Date'])
    
    #District of the current market
    district = cur_df['District'].unique()[0]
    
    # Iterating over all dates from 2009-18
    for date in dates:
        # if a date is not available in considered market data
        if date not in cur_dates:
            # add dummy entry with price and arrival as NAN
            new_df_list.append( [district, market,math.nan,date
                                     ,math.nan])
        else:
            count +=1
            
# Converting new_df_list to dataframe
neww_df = pd.DataFrame(new_df_list, columns=['District', 'Market', 'Arrivals', 'Date', 'Modal_Price'])

#Dropping duplicates
neww_df = neww_df.drop_duplicates(
  subset = ['Market', 'Date'],
  keep = 'first').reset_index(drop = True)

#appending with actual data
mod_df_ = mod_df.append(neww_df)
            
mod_df_ = mod_df_.drop_duplicates(
  subset = ['Market', 'Date'],
  keep = 'first').reset_index(drop = True)



# adding month, month name , day, year column
df=mod_df_

month_array = []
day_array=[]
year_array=[]
for i in range(df.shape[0]):
    month_array.append(df["Date"][i][3:6])
    day_array.append(int(df['Date'][i][0:2]))
    year_array.append(int(df['Date'][i][7:]))
    
df['Month_name']= month_array
df['Day'] = day_array
df['Year']= year_array

month_names = np.array(df['Month_name'])
month_names = list(month_names) 
months = month_names.copy()

for i in range(len(months)):
  if month_names[i] == "Jan":
    months[i] = 1
  elif month_names[i] == "Feb":
    months[i] = 2
  elif month_names[i] == "Mar":
    months[i] = 3
  elif month_names[i] == "Apr":
    months[i] = 4
  elif month_names[i] == "May":
    months[i] = 5
  elif month_names[i] == "Jun":
    months[i] = 6
  elif month_names[i] == "Jul":
    months[i] = 7
  elif month_names[i] == "Aug":
    months[i] = 8
  elif month_names[i] == "Sep":
    months[i] = 9
  elif month_names[i] == "Oct":
    months[i] = 10
  elif month_names[i] == "Nov":
    months[i] = 11
  elif month_names[i] == "Dec":
    months[i] = 12

df['Month']=months

mod_df_ = df

#Sorting data
mod_df_ = mod_df_.sort_values(by = ['District','Market','Year','Month','Day'])
mod_df_ = mod_df_.reset_index(drop=True)
    
    
# taking price and arrival data into array
arr1 = mod_df_['Modal Price'].to_numpy()
arr2 = mod_df_['Arrivals'].to_numpy()

# Replacing 0 with NAN 
arr1[arr1 == 0] = 'nan'
arr2[arr2 == 0] = 'nan'

total_days = 3652 # from 01-01-2009 to 31-12-2018
arr1 = arr1.reshape(len(mandi_list),total_days)
arr2 = arr2.reshape(len(mandi_list),total_days)
arr1.shape,arr2.shape



#Imputation

#using knn
X_filled_knn = KNN(k=3).fit_transform(arr1)
X_filled_knn_arrival = KNN(k=3).fit_transform(arr2)

#Filling 0 (if any) with the previous available price or arrival for the market
# if previously data is not available, then replacing with nearby market same date data

for i in range(X_filled_knn.shape[0]):
    for j in range(X_filled_knn.shape[1]):
        if X_filled_knn[i][j] == 0:
            if j >0:
                X_filled_knn[i][j]= X_filled_knn[i][j-1]
            else:
                for k in X_filled_knn[:,j]:
                    if k !=0:
                        i[index]= k
                        break
                    
for i in range(X_filled_knn_arrival.shape[0]):
    for j in range(X_filled_knn_arrival.shape[1]):
        if X_filled_knn_arrival[i][j] == 0:
            if j >0:
                X_filled_knn_arrival[i][j]= X_filled_knn_arrival[i][j-1]
            else:
                for k in X_filled_knn_arrival[:,j]:
                    if k !=0:
                        i[index]= k
                        break

                        
# SoftImpute

X_filled_softimpute = SoftImpute().fit_transform(arr1)

X_filled_softimpute_arrival = SoftImpute().fit_transform(arr2)

for i in range(X_filled_softimpute_arrival.shape[0]):
    for j in range(X_filled_softimpute_arrival.shape[1]):
        if X_filled_softimpute_arrival[i][j] == 0:
            if j >0:
                X_filled_softimpute_arrival[i][j]= X_filled_softimpute_arrival[i][j-1]
            else:
                for k in X_filled_softimpute_arrival[:,j]:
                    if k !=0:
                        i[index]= k
                        break
                        
                        
for i in range(X_filled_softimpute.shape[0]):
    for j in range(X_filled_softimpute.shape[1]):
        if X_filled_softimpute[i][j] == 0:
            if j >0:
                X_filled_softimpute[i][j]= X_filled_softimpute[i][j-1]
            else:
                for k in X_filled_softimpute[:,j]:
                    if k !=0:
                        i[index]= k
                        break
                        
                        

# Taking average of both imputation technique

X_filled_avg_SI_Knn = np.full((X_filled_softimpute.shape[0],X_filled_softimpute.shape[1]),0)

for i in range(X_filled_softimpute.shape[0]):
    for j in range(X_filled_softimpute.shape[1]):
        
        X_filled_avg_SI_Knn[i][j] = (X_filled_softimpute[i][j]+X_filled_knn[i][j])/2

X_filled_avg_SI_Knn_arrival = np.full((X_filled_softimpute_arrival.shape[0],X_filled_softimpute_arrival.shape[1]),0)

for i in range(X_filled_softimpute_arrival.shape[0]):
    for j in range(X_filled_softimpute_arrival.shape[1]):
        
        X_filled_avg_SI_Knn_arrival[i][j] = (X_filled_softimpute_arrival[i][j]+X_filled_knn_arrival[i][j])/2
        
        
# Saving imputed array
np.save('Wheat_Updated_SIKNN_price_post_234_market_imp_v4.npy', X_filled_avg_SI_Knn)
np.save('Wheat_Updated_SIKNN_ARRIVAL_post_234_market_imp_v4.npy', X_filled_avg_SI_Knn_arrival)

#Updating data frame with imputed values
mod_df_['Imp_Price'] = X_filled_avg_SI_Knn.reshape(-1,1)
mod_df_['Imp_Arrival'] = X_filled_avg_SI_Knn_arrival.reshape(-1,1)
# adding flag whether the data was available(True) or imputed(False)
ArrivalNanFlag = np.isnan(mod_df_['Arrivals'])
PriceNanFlag = np.isnan(mod_df_['Modal Price'])
mod_df_['PriceNanFlag'] = PriceNanFlag
mod_df_['ArrivalNanFlag'] = ArrivalNanFlag


# Dropping columns
mod_df_ =mod_df_.drop(['Modal Price', 'Modal_Price','Arrivals'],axis=1)

# Saving dataframe into csv format
mod_df_.to_csv('Wheat/Wheat/Wheat_UPDATED_234mandis_post_impute.csv')

df=mod_df_
# Sanity check on imputation

# if imputed values are not between the past and future available data then we are replacing it with mean of past and future data
def fill_misses(n,check=False):
    # n is the number of continous misses in the data which are imputed now
    
    i=1
    count=0
    previous_missing=False
    next_missing = False
    current_missing= False
    while(i<df.shape[0]-n):
        flag=True
        for j in range(n):
            flag= flag and df['ArrivalNanFlag'][i+j]
        current_missing = flag
        previous_missing = df['ArrivalNanFlag'][i-1]
        next_missing = df['ArrivalNanFlag'][i+n]
        
#         print('d')
        if current_missing==True and previous_missing==False and next_missing==False:
            for j in range(n):
                if check:
                   
                    if (df['Imp_Price'][i+j]<df['Imp_Price'][i-1] and df['Imp_Price'][i+j]>df['Imp_Price'][i+n]) or (df['Imp_Price'][i+j]>df['Imp_Price'][i-1] and df['Imp_Price'][i+j]<df['Imp_Price'][i+n]) == False:
                        df['Imp_Price'][i+j] = (df['Imp_Price'][i-1]+df['Imp_Price'][i+n])/2
                        df['Imp_Arrival'][i+j] = (df['Imp_Arrival'][i-1]+df['Imp_Arrival'][i+n])/2
                else:    
                    df['Imp_Price'][i+j] = (df['Imp_Price'][i-1]+df['Imp_Price'][i+n])/2
                    df['Imp_Arrival'][i+j] = (df['Imp_Arrival'][i-1]+df['Imp_Arrival'][i+n])/2

            count+=1
        i+=1
    return count




fill_misses(1)
fill_misses(2)
fill_misses(4)
fill_misses(5)
fill_misses(6)

df.to_csv('Wheat/Wheat/Wheat__UPDATED_all_mandis_post_impute.csv')

