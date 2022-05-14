# Reading csv into dataframe
df = pd.read_csv('Tomato/Tomato/sample_post_impute.csv')


# Adding new columns as month namem, month, day, year from column date
month_array = []
day_array=[]
year_array=[]
for i in range(df.shape[0]):
    month_array.append(df["Date"][i][3:6])
    day_array.append(df['Date'][i][0:2])
    year_array.append(df['Date'][i][7:])
    
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


# Drop unneccessary columns and sorting dataframe 
df= df.drop(['Unnamed: 0'], axis=1)
df = df.sort_values(by = ['District','Market', 'Year','Month','Day'])
df = df.reset_index(drop=True)


# latitude and longitude for 20 mandis
latitude_20_markets = [('Rampur', 24.890090999999998),
             ('Muradabad', 28.8334982),
             ('Bareilly', 28.457876),
             ('Lakhimpur', 27.985060150000002),
             ('Basti', 26.724789),
             ('Shahjahanpur', 27.912633149999998),
             ('Faizabad', 26.63807555),
             ('Raibareilly', 26.230299),
             ('Lucknow', 26.8381),
             ('Aligarh', 27.87698975),
             ('Kasganj', 27.883846050000002),
             ('Bijnaur', 29.8575065),
             ('Ballia', 25.877932549999997),
             ('Muzzafarnagar', 29.4115745),
             ('Unnao', 26.57550365),
             ('Gazipur', 25.603508400000003),
             ('Gorakhpur', 26.6711433),
             ('Jaunpur', 25.7955927),
             ('Sultanpur', 26.242510850000002),
             ('Bahraich', 27.7336958)]
longitude_20_markets = [('Rampur', 83.73254274787365),
             ('Muradabad', 78.7732864),
             ('Bareilly', 79.40557093743058),
             ('Lakhimpur', 80.75384538357649),
             ('Basti', 82.79326865024002),
             ('Shahjahanpur', 79.74656294869826),
             ('Faizabad', 82.05902434378625),
             ('Raibareilly', 81.240891),
             ('Lucknow', 80.9346001),
             ('Aligarh', 78.13729027600994),
             ('Kasganj', 78.63489003747873),
             ('Bijnaur', 78.5598995),
             ('Ballia', 84.11995931460379),
             ('Muzzafarnagar', 77.7698696),
             ('Unnao', 80.61376177782856),
             ('Gazipur', 83.50745404887138),
             ('Gorakhpur', 83.36457243864551),
             ('Jaunpur', 82.48834097504385),
             ('Sultanpur', 82.29616931685918),
             ('Bahraich', 81.47732127661058)]


# taking only data of above 20 mandis 

top_20_markets = [i[0] for i in longitude_20_markets]
df = df_new.loc[df_new['Market'].isin(top_20_markets)]
df = df.reset_index(drop=True)

# making dictionaries for lat long of every market

dict_lat ={}
for i in latitude_20_markets:
    dict_lat[i[0]]=i[1]
    
dict_long ={}
for i in longitude_20_markets:
    dict_long[i[0]]=i[1]
    
long_list=[]
lat_list=[]

for i in range(df.shape[0]):
    market = df['Market'][i]
    long_list.append(dict_long[market])
    lat_list.append(dict_lat[market])
    
    
# Adding lat long to dataframe    
df['latitude']=lat_list
df['longitude']=long_list





# Reading environmental data

df_env = pd.read_csv('Environmental_data_date_wise.csv')
df_env= df_env.drop(['Unnamed: 0'], axis=1)

month_dict = {'01':'Jan','02':'Feb', '03':'Mar','04':'Apr','05': 'May','06': 'Jun','07':'Jul','08': 'Aug','09':'Sep','10': 'Oct','11':'Nov','12':'Dec'}

# adding day, month,month name, year columns similar to ones in price data
dates=[]
month_names=[]
years=[]
month=[]

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



#lat long of markets are like 79.8163 28.028, but we have environmental data for lat long 80.0 and 28.0
# Therfore making market's lat long to the nearest lat long available in environmental dataset

target_lat= df_env['latitude'].unique()
target_long=df_env['longitude'].unique()
lat=df['latitude'].unique()
long=df['longitude'].unique()



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
df.head()


df= df.drop(['latitude_env','longitude_env'], axis=1)

# Merging price data with env data
df_new = pd.merge(df, df_env, on=['Date','latitude','longitude'])


df_new = df_new.sort_values(by = ['District','Market', 'Year','Month','Day'])
df_new = df_new.reset_index(drop=True)


df_new.to_csv('Tomato/Tomato/sample_env_post_impute.csv')
