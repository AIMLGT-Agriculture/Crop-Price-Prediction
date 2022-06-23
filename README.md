# Crop-Price-Prediction
Predict the upcoming crop prices using crop's historical data and other crop related parameters such as rainfall, geographical location, sunlight, humidity, etc.
![Image1](https://github.com/AIMLGT-Agriculture/Crop-Price-Prediction/blob/main/act_vs_pred.png) 
Actual vs Predicted normalized price

We have made prediction for wheat, potato, onion, chilli, brinjal, and tomato for upcoming 4, 6, 9, and 15 days.

## Motivation
* Agriculture plays a vital role in Indian economy
* It provides employment to over 60% of the population
* Farmers struggle to make profit
* Lack of knowledge about expected prices is the key reason for their struggle

## Use-Cases
Price prediction can help Government to intervene to support farmers

It helps farmers decide:

* Crop selection
* When to sow
* When to harvest
* When to sell

## Factors Affecting Crop Prices

Historical data
1. Price
2. Arrival

Environmental factors
1. Rainfall
2. Location
3. Temperature
4. Humidity, etc

Demand-side factors
1. Crop's Demand
2. Import/Export
3. Inflation/Agflation
4. Population Change

## Novel Contribution
We have proposed a GNN+LSTM model (inspired from [4]) to exploit the Spatio-temporal in the data accross nearby markets.

![Image2](https://github.com/AIMLGT-Agriculture/Crop-Price-Prediction/blob/main/GNN_MODEL.png)

* GNN model tries to capture different relations among nearby markets
* LSTM tries to capture and memorize the various trends in historical price and other datasets.

## Experimental Results

### EDA
![Image3](https://github.com/AIMLGT-Agriculture/Crop-Price-Prediction/blob/main/UP_monthly_avg_price.png)
![Image4](https://github.com/AIMLGT-Agriculture/Crop-Price-Prediction/blob/main/yearly_avg_price.png)
![Image5](https://github.com/AIMLGT-Agriculture/Crop-Price-Prediction/blob/main/avg_price_vs_markets%20wheat.png)
![Image6](https://github.com/AIMLGT-Agriculture/Crop-Price-Prediction/blob/main/prices%20for%20diff%20markets.png)

### Outperforming state-of-the-art
Our model outperforms state-of-the-art PECAD model[3]
![Image2](https://github.com/AIMLGT-Agriculture/Crop-Price-Prediction/blob/main/cov4,6,9%20Chilli.png)
![Image2](https://github.com/AIMLGT-Agriculture/Crop-Price-Prediction/blob/main/cov4,6,9%20tomato.png)
![Image2](https://github.com/AIMLGT-Agriculture/Crop-Price-Prediction/blob/main/cropvsdaysRMSE.png)
![Image2](https://github.com/AIMLGT-Agriculture/Crop-Price-Prediction/blob/main/exp_res_table1_2.png)
![Image2](https://github.com/AIMLGT-Agriculture/Crop-Price-Prediction/blob/main/exp_res_table3_4.png)
![Image2](https://github.com/AIMLGT-Agriculture/Crop-Price-Prediction/blob/main/exp_res_table5_6.png)

## References 
[1] Data available at agmarknet.gov.in

[2] Data available at copernicus.eu

[3] Guo, H., Woodruff, A., & Yadav, A. (2020). Improving Lives of Indebted Farmers Using Deep Learning: Predicting Agricultural Produce Prices
Using Convolutional Neural Networks. Proceedings of the AAAI Conference on Artificial Intelligence, 34(08), 13294-13299.
https://doi.org/10.1609/aaai.v34i08.7039

[4] Bing Yu, Haoteng Yin, and Zhanxing Zhu. Spatio-temporal graph convolutional neural network: A deep learning framework for traffic forecasting. CoRR, abs/1709.04875, 2017.


# Code flow
1. Impute the price and arrival data          # Use Softimpute_KNN_imputation.py
2. Add environmental data to price and arrival data  # Use add_weather_data.py
3. Run GNN+LSTM model to make prediction and get performance metrics  # GNN+LSTM_model.py

All the required data is available here - https://indianinstituteofscience-my.sharepoint.com/:f:/g/personal/deepanshu1_iisc_ac_in/Evcc_sCU5OBHuSGzy-x2nroBZVwx1oc7WrwIya6osGMc-g?e=0RMVUd

## 1. Softimpute_KNN_imputation.py
#### This python file takes raw price and arrival data downloaded from agmarknet.gov.in and do following:

a) Merge price and arrival based on the dates 

b) Impute missing values for price and arrival using KNN and softimpute

c) Update imputed values if they are different from the past and future available values

d) Save the imputed file


#### To run this file do following:

i. Download raw price and arrival data from the given onedrive link

ii. Update price and arrival data locations according to your downloaded location in file where we are reading these two files in dataframe

iii. Update the saving location for files

iv. Run the python file using command python3 Softimpute+KNN_imputation.py

## 2. add_weather_data.py

#### This file adds environmental data to the imputed file obtained using Softimpute+KNN_imputation.py

#### To run this file do following:

i. Update the address of imputed file with the location of the imputed file received from running file Softimpute+KNN_imputation.py

ii. Run file using python3 add_weather_data.py

This file will save the price and arrival added with environmental data. Update the saving location in file 


## 3. GNN+LSTM_model.py
#### This file run GNN+LSTM model on the data obtained from running add_weather_data.py

#### To run this file do following:

i. Update the address of csv file with the location of the csv file downloaded from running file add_weather_data.py

ii. Change the value of variable days according to number of days you want prediction for

iii. Run file using python3 GNN+LSTM_model.py


## Earlier code description below

## 15 days prediction
lstms_15days_20mandis.py predicts prices for net 15 days by looking the data of last 15 days for 20 mandis.

Output is normalized before feeding into lstms model to obtain better results.

Used different embededings. Month and dates - cyclic embedding ; markets - one hot vector and lat long

Updated lat long values with market locations which was for districts earlier


## Prediction for up to next 7 days using LSTMs
crop_price_prediction_lstm_7_days.py file predicts prices for next 7 days by looking at the data of last 10 days.

On running the above file you can either choose to run the trained model to see the performance metrics on test data and sample predictions or you can train the model and then see the results.

As of now model parameters are already given in the file, you can change them manually by editing in the .py file.

## Predicition for next day using random forests and lasso regression
price_prediction_website_30.py file predicts the next day prices.

It gives user option to either train the model or run the already trained model.
It also provides option to eliminate some features if training the model. Moreover, one can select a market and date to see the next day prediction.
