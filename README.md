# Crop-Price-Prediction

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
