#Importing required files

import pandas as pd
import numpy as np
import os
import typing
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import metrics
from scipy.stats import pearsonr
import statistics
from sklearn.metrics import r2_score

import geopy.distance
import googlemaps  
from tensorflow.keras.preprocessing import timeseries_dataset_from_array



class GraphInfo:
    def __init__(self, edges: typing.Tuple[list, list], num_nodes: int):
        self.edges = edges
        self.num_nodes = num_nodes





class GraphConv(layers.Layer):
    def __init__(
        self,
        in_feat,
        out_feat,
        graph_info: GraphInfo,
        aggregation_type="mean",
        combination_type="concat",
        activation: typing.Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.graph_info = graph_info
        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.weight = tf.Variable(
            initial_value=keras.initializers.glorot_uniform()(
                shape=(in_feat, out_feat), dtype="float32"
            ),
            trainable=True,
        )
        self.activation = layers.Activation(activation)

    def aggregate(self, neighbour_representations: tf.Tensor):
        aggregation_func = {
            "sum": tf.math.unsorted_segment_sum,
            "mean": tf.math.unsorted_segment_mean,
            "max": tf.math.unsorted_segment_max,
        }.get(self.aggregation_type)

        if aggregation_func:
            return aggregation_func(
                neighbour_representations,
                self.graph_info.edges[0],
                num_segments=self.graph_info.num_nodes,
            )

        raise ValueError(f"Invalid aggregation type: {self.aggregation_type}")

    def compute_nodes_representation(self, features: tf.Tensor):
        """Computes each node's representation.

        The nodes' representations are obtained by multiplying the features tensor with
        `self.weight`. Note that
        `self.weight` has shape `(in_feat, out_feat)`.

        Args:
            features: Tensor of shape `(num_nodes, batch_size, input_seq_len, in_feat)`

        Returns:
            A tensor of shape `(num_nodes, batch_size, input_seq_len, out_feat)`
        """
        return tf.matmul(features, self.weight)

    def compute_aggregated_messages(self, features: tf.Tensor):
        neighbour_representations = tf.gather(features, self.graph_info.edges[1])
        aggregated_messages = self.aggregate(neighbour_representations)
        return tf.matmul(aggregated_messages, self.weight)

    def update(self, nodes_representation: tf.Tensor, aggregated_messages: tf.Tensor):
        if self.combination_type == "concat":
            h = tf.concat([nodes_representation, aggregated_messages], axis=-1)
        elif self.combination_type == "add":
            h = nodes_representation + aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}.")

        return self.activation(h)

    def call(self, features: tf.Tensor):
        """Forward pass.

        Args:
            features: tensor of shape `(num_nodes, batch_size, input_seq_len, in_feat)`

        Returns:
            A tensor of shape `(num_nodes, batch_size, input_seq_len, out_feat)`
        """
        nodes_representation = self.compute_nodes_representation(features)
        aggregated_messages = self.compute_aggregated_messages(features)
        return self.update(nodes_representation, aggregated_messages)



        


def preprocess(data_array: np.ndarray, train_size: float, val_size: float):
    """Splits data into train/val/test sets and normalizes the data.

    Args:
        data_array: ndarray of shape `(num_time_steps, num_routes)`
        train_size: A float value between 0.0 and 1.0 that represent the proportion of the dataset
            to include in the train split.
        val_size: A float value between 0.0 and 1.0 that represent the proportion of the dataset
            to include in the validation split.

    Returns:
        `train_array`, `val_array`, `test_array`
    """

    num_time_steps = data_array.shape[0]
    num_train, num_val = (
        int(num_time_steps * train_size),
        int(num_time_steps * val_size),
    )
    train_array = data_array[:num_train]
    mean, std = train_array.mean(axis=0), train_array.std(axis=0)

    train_array = (train_array - mean) / std
    val_array = (data_array[num_train : (num_train + num_val)] - mean) / std
    test_array = (data_array[(num_train + num_val) :] - mean) / std

    return train_array, val_array, test_array,mean,std



def create_tf_dataset(
    data_array: np.ndarray,
    input_sequence_length: int,
    forecast_horizon: int,
    batch_size: int = 128,
    shuffle=True,
    multi_horizon=True,
):
    """Creates tensorflow dataset from numpy array.

    This function creates a dataset where each element is a tuple `(inputs, targets)`.
    `inputs` is a Tensor
    of shape `(batch_size, input_sequence_length, num_routes, 1)` containing
    the `input_sequence_length` past values of the timeseries for each node.
    `targets` is a Tensor of shape `(batch_size, forecast_horizon, num_routes)`
    containing the `forecast_horizon`
    future values of the timeseries for each node.

    Args:
        data_array: np.ndarray with shape `(num_time_steps, num_routes)`
        input_sequence_length: Length of the input sequence (in number of timesteps).
        forecast_horizon: If `multi_horizon=True`, the target will be the values of the timeseries for 1 to
            `forecast_horizon` timesteps ahead. If `multi_horizon=False`, the target will be the value of the
            timeseries `forecast_horizon` steps ahead (only one value).
        batch_size: Number of timeseries samples in each batch.
        shuffle: Whether to shuffle output samples, or instead draw them in chronological order.
        multi_horizon: See `forecast_horizon`.

    Returns:
        A tf.data.Dataset instance.
    """

    inputs = timeseries_dataset_from_array(
        np.expand_dims(data_array[:-forecast_horizon], axis=-1),
        None,
        sequence_length=input_sequence_length,
        shuffle=False,
        batch_size=batch_size,
    )

    target_offset = (
        input_sequence_length
        if multi_horizon
        else input_sequence_length + forecast_horizon - 1
    )
    target_seq_length = forecast_horizon if multi_horizon else 1
    targets = timeseries_dataset_from_array(
        data_array[target_offset:],
        None,
        sequence_length=target_seq_length,
        shuffle=False,
        batch_size=batch_size,
    )

    dataset = tf.data.Dataset.zip((inputs, targets))
    if shuffle:
        dataset = dataset.shuffle(100)

    return dataset.prefetch(16).cache()



#https://medium.com/how-to-use-google-distance-matrix-api-in-python/how-to-use-google-distance-matrix-api-in-python-ef9cd895303c

def distance(m1, m2, loc_dict):
  d=0
  c1=(loc_dict[m1]['latitude'],loc_dict[m1]['longitude'])
  c2=(loc_dict[m2]['latitude'],loc_dict[m2]['longitude'])
  return geopy.distance.vincenty(c1,c2).km

def google_distance(m1, m2, loc_dict):
  gmaps = googlemaps.Client(key='Your_API_key')
  origin_latitude = loc_dict[m1]['latitude']
  origin_longitude = loc_dict[m1]['longitude']
  destination_latitude = loc_dict[m2]['latitude']
  destination_longitude = loc_dict[m2]['longitude']
  return gmaps.distance_matrix([str(origin_latitude) + " " + str(origin_longitude)], [str(destination_latitude) + " " + str(destination_longitude)], mode='walking')['rows'][0]['elements'][0]



def adj(threshold, names, loc_dict,markets):
    ''' 
    return adj matrix
    '''
  markets = markets
  adj_mat = np.full((len(markets),len(markets)),0)
  for i in range(len(markets)):
    for j in range(len(markets)):
      if distance(markets[i], markets[j], loc_dict)<threshold:
        adj_mat[i][j]=1

  return adj_mat




#Reading data into dataframe
df = pd.read_csv('/content/drive/MyDrive/environmental_data/Wheat__UPDATED_all_mandis_env_post_impute.csv')

# Reading price data into list

n_markets = len(df['Market'].unique())

#Reshaping price data into 2d array
price_array = price_array.reshape((254,-1))
price_array = price_array.T

# Now price array is in the 2d array form having dimensions as (number of dates * number of markets)


###########################
# Making Adj matrix based on the distance
# In our graph every node represents a market 
# If two markets are within 200kms then there will be an edge between them
##########################


#Storing lat long for all the markets
lat_dict={}             # latitude dictionary
long_dict={}            # longitude dictionary

for i in range(df.shape[0]):
  market = df['Market'][i]
  lat_dict[market]=df['latitude'][i]
  long_dict[market]=df['longitude'][i]



loc_dict={}            # location dictionary consisting of lat long for every market

for  mandi,lat in lat_dict.items():
  loc_dict[mandi]={}
  loc_dict[mandi]['latitude']=lat

for mandi,lon in long_dict.items():
  loc_dict[mandi]['longitude']=lon



#Making adj matrix
route_dist = adj(200,False,loc_dict,markets)

# train data - 2009 to 2015, val data - 2016 and test data- 2017 to 2018
train_size, val_size = 0.7, 0.1
train_array, val_array, test_array,mean,std = preprocess(price_array, train_size, val_size)


#Number of days price to be forcasted for in future
days =15


batch_size = 64
input_sequence_length = 15
forecast_horizon = days
multi_horizon = True


#Creating train and val data
train_dataset, val_dataset = (
    create_tf_dataset(data_array, input_sequence_length, forecast_horizon, batch_size)
    for data_array in [train_array, val_array]
)

#creating test data
test_dataset = create_tf_dataset(
    test_array,
    input_sequence_length,
    forecast_horizon,
    batch_size=test_array.shape[0],
    shuffle=False,
    multi_horizon=multi_horizon,
)


#Making graph

adjacency_matrix = route_dist
node_indices, neighbor_indices = np.where(adjacency_matrix == 1)
graph = GraphInfo(
    edges=(node_indices.tolist(), neighbor_indices.tolist()),
    num_nodes=adjacency_matrix.shape[0],
)



in_feat = 1
batch_size = 64
epochs = 70
input_sequence_length = 15
forecast_horizon = days
multi_horizon = True
out_feat = 10
lstm_units = 128
graph_conv_params = {
    "aggregation_type": "mean",
    "combination_type": "concat",
    "activation": None,
}
#Defining model
st_gcn = LSTMGC(
    in_feat,
    out_feat,
    lstm_units,
    input_sequence_length,
    forecast_horizon,
    graph,
    graph_conv_params,
)
inputs = layers.Input((input_sequence_length, graph.num_nodes, in_feat))
outputs = st_gcn(inputs)

model = keras.models.Model(inputs, outputs)
# Defining loss and optimizer
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=0.0002),
    loss=keras.losses.MeanSquaredError(),
)

#Model training
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=[keras.callbacks.EarlyStopping(patience=10)],
)



x_test, y = next(test_dataset.as_numpy_iterator())
#Making predictions
y_pred = model.predict(x_test)


# Calculating performance matrics
rmse_list=[]
mae_list=[]
actual_price=[]
predicted_price=[]
for i in range(y_pred.shape[2]):
  normalized_market_price = y_pred[:,:,i]
  market_price = normalized_market_price*std[i] + mean[i]
  predicted_price.append(market_price)

  actual_normalized_market_price = y[:,:,i]
  actual_market_price = actual_normalized_market_price*std[i] + mean[i]
  actual_price.append(actual_market_price)

  rmse = np.sqrt(metrics.mean_squared_error(actual_market_price, market_price))
  mae = metrics.mean_absolute_error(actual_market_price, market_price)

  rmse_list.append(rmse)
  mae_list.append(mae)

rmse_list_array = np.array(rmse_list)
# np.save('/content/drive/MyDrive/environmental_data/rmse_list_array_wheat_15.npy',rmse_list_array)

rmse=sum(rmse_list)/len(rmse_list)
mae=sum(mae_list)/len(mae_list)

actual_price = np.array(actual_price)
predicted_price = np.array(predicted_price)


c=actual_price.reshape(-1,1)
d=predicted_price.reshape(-1,1)

c = list(c)
d=list(d)

list1 = [int(i) for i in c]
list2 = [int(i) for i in d]

std_=statistics.stdev(list2)

corr, _ = pearsonr(list1, list2)
r2 = r2_score(list1, list2)
print('MAE: ',mae)
print('RMSE : ', rmse)
print('Normalized RMSE with max min : ', rmse/(max(d)-min(d)))
print('Coeff of variation : ', std_/np.mean(d))
print('R squared: ',r2)
print('Pearson corr: ',corr)
print('Normalized RMSE with mean : ', rmse/np.mean(price_array.reshape(-1,1)))

