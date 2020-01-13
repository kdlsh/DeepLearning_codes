# https://www.tensorflow.org/tutorials/structured_data/time_series

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os, sys, re
import pandas as pd

# os.chdir(os.path.dirname(os.path.realpath(__file__)))

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

## The dataset
txt_path = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\stacked_data.csv"
#df = pd.read_csv(txt_path, sep='\t', lineterminator='\r')
df = pd.read_csv(txt_path)
print(df.head())
df = df.drop(['Unsold','Completed','Starts','JSratio','JW'], axis=1)
df = df.dropna()
print(df.head())


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)


## fisrt 100 rows : training dataset, remaining : validation dataset
TRAIN_SPLIT = 110
tf.random.set_seed(10)

past_history = 12  #24
future_target = 6 #12
STEP = 1

## Multi-Step model (predict a sequence of the future)
def build_multi_step_train_val_data(df):
    features_considered = ['JS', 'MM', 'Permits']
    #features_considered = ['MM', 'Permits']

    features = df[features_considered]
    features.index = df['Date Time']
    features.head()

    #features.plot(subplots=True)

    dataset = features.values
    data_mean = dataset.mean(axis=0)
    data_std = dataset.std(axis=0)

    #dataset = (dataset-data_mean)/data_std

    # past_history = 24
    # future_target = 6
    # STEP = 1
    # dataset[:, 1] -> y_train_multi
    
    x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 1], 0,
                                                    TRAIN_SPLIT, past_history,
                                                    future_target, STEP)
    x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 1],
                                                TRAIN_SPLIT, None, past_history,
                                                future_target, STEP)
                                            
    return x_train_multi, y_train_multi, x_val_multi, y_val_multi
                                            
x_train_li = []
y_train_li = []
x_val_li = []
y_val_li = []
for region in list(df['Reg'].drop_duplicates()):
    region_df = df.loc[df['Reg'] == region]
    x_t, y_t, x_v, y_v = build_multi_step_train_val_data(region_df)
    x_train_li.append(x_t)
    y_train_li.append(y_t)
    x_val_li.append(x_v)
    y_val_li.append(y_v)

x_train_multi = np.concatenate(x_train_li, axis=0)
y_train_multi = np.concatenate(y_train_li, axis=0)
x_val_multi = np.concatenate(x_val_li, axis=0)
y_val_multi = np.concatenate(y_val_li, axis=0)

print (x_train_multi.shape)
print (y_train_multi.shape)
print (x_val_multi.shape)
print (y_val_multi.shape)
print ('Single window of past history : {}'.format(x_train_multi[0].shape))
print ('\n Target temperature to predict : {}'.format(y_train_multi[0].shape))


## Recurrent neural network
BATCH_SIZE = 192
BUFFER_SIZE = 500

train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()


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
    
def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()

def multi_step_plot(history, true_future, prediction):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, 1]), label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
            label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
                label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()

for x, y in train_data_multi.take(1):
    multi_step_plot(x[0], y[0], np.array([0]))

multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(12,
                                          return_sequences=True,
                                          input_shape=x_train_multi.shape[-2:]))
multi_step_model.add(tf.keras.layers.LSTM(8, activation='relu'))
multi_step_model.add(tf.keras.layers.Dense(future_target))

multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

for x, y in val_data_multi.take(1):
    print (multi_step_model.predict(x).shape)

EVALUATION_INTERVAL = 150
EPOCHS = 15

multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data_multi,
                                          validation_steps=50)
                                    
plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')

for x, y in val_data_multi.take(5):
    multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])

## Time series windowing
# https://www.tensorflow.org/guide/data#time_series_windowing