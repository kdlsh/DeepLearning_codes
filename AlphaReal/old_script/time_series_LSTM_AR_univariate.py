# https://www.tensorflow.org/tutorials/structured_data/time_series

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# os.chdir(os.path.dirname(os.path.realpath(__file__)))

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

## The dataset
txt_path = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\data.txt"
#df = pd.read_csv(txt_path, sep='\t', lineterminator='\r')
df = pd.read_csv(txt_path, sep='\t')
print(df.head())
df = df.drop(['Unsold','Completed','Starts'], axis=1)
print(df.head())


def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i+target_size])
    return np.array(data), np.array(labels)


## fisrt 100 rows : training dataset, remaining : validation dataset
TRAIN_SPLIT = 120
tf.random.set_seed(10)

## Part 1: Forecast a univariate time series
def build_train_val_data(df):
    uni_data = df['MM']
    uni_data.index = df['Date Time']
    #uni_data.head()

    #uni_data.plot(subplots=True)
    #uni_data.describe()

    uni_data = uni_data.values
    uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
    uni_train_std = uni_data[:TRAIN_SPLIT].std()

    uni_data = (uni_data-uni_train_mean)/uni_train_std

    ## 24 month history prediction
    univariate_past_history = 24
    univariate_future_target = 0

    x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT, 
                                                univariate_past_history,
                                                univariate_future_target)                                           
    x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None, 
                                                univariate_past_history, 
                                                univariate_future_target)
                                  
    return x_train_uni, y_train_uni, x_val_uni, y_val_uni

x_train_uni_li = []
y_train_uni_li = []
x_val_uni_li = []
y_val_uni_li = []
for region in list(df['Reg'].drop_duplicates()):
    region_df = df.loc[df['Reg'] == region]
    x_t, y_t, x_v, y_v = build_train_val_data(region_df)
    x_train_uni_li.append(x_t)
    y_train_uni_li.append(y_t)
    x_val_uni_li.append(x_v)
    y_val_uni_li.append(y_v)

x_train_uni = np.concatenate(x_train_uni_li, axis=0)
y_train_uni = np.concatenate(y_train_uni_li, axis=0)
x_val_uni = np.concatenate(x_val_uni_li, axis=0)
y_val_uni = np.concatenate(y_val_uni_li, axis=0)

print ('Single window of past history')
print (x_train_uni.shape)
print (x_train_uni[0])
print ('\n Target temperature to predict')
print (y_train_uni.shape)
print (y_train_uni[0])
print (x_val_uni.shape)
print (y_val_uni.shape)

def create_time_steps(length):
    time_steps = []
    for i in range(-length, 0, 1):
        time_steps.append(i)
    return time_steps

def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    plt.show()
    return plt

# show_plot([x_train_uni[0], y_train_uni[0]], 0, 'Sample Example')

## Baseline (mean)
def baseline(history):
    return np.mean(history)

# show_plot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])], 0,
#            'Baseline Prediction Example')

## Recurrent neural network
BATCH_SIZE = 96
BUFFER_SIZE = 400

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(10, input_shape=x_train_uni.shape[-2:]),
    tf.keras.layers.Dense(1)
])

simple_lstm_model.compile(optimizer='adam', loss='mae')

for x, y in val_univariate.take(1):
    print(simple_lstm_model.predict(x).shape)

EVALUATION_INTERVAL = 200
EPOCHS = 30

simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val_univariate, validation_steps=50)


## Predict using the simple LSTM model
# for x, y in val_univariate.take(3):
#     plot = show_plot([x[0].numpy(), y[0].numpy(),
#                         simple_lstm_model.predict(x)[0]], 0, 'Simple LSTM model')
#     plot.show()


