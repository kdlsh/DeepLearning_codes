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
TRAIN_SPLIT = 120
tf.random.set_seed(13)

## Forecast a multivariate time series
def build_single_step_train_val_data(df):
    features_considered = ['JS', 'MM', 'Permits']

    features = df[features_considered]
    features.index = df['Date Time']
    features.head()

    #features.plot(subplots=True)

    dataset = features.values
    data_mean = dataset.mean(axis=0)
    data_std = dataset.std(axis=0)

    dataset = (dataset-data_mean)/data_std

    ## Single step model
    past_history = 24
    future_target = 6
    STEP = 1

    x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 1], 0,
                                                    TRAIN_SPLIT, past_history,
                                                    future_target, STEP,
                                                    single_step=True)
    x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 1],
                                                TRAIN_SPLIT, None, past_history,
                                                future_target, STEP,
                                                single_step=True)
                                            
    return x_train_single, y_train_single, x_val_single, y_val_single
                                            
x_train_single_li = []
y_train_single_li = []
x_val_single_li = []
y_val_single_li = []
for region in list(df['Reg'].drop_duplicates()):
    region_df = df.loc[df['Reg'] == region]
    x_t, y_t, x_v, y_v = build_single_step_train_val_data(region_df)
    x_train_single_li.append(x_t)
    y_train_single_li.append(y_t)
    x_val_single_li.append(x_v)
    y_val_single_li.append(y_v)

x_train_single = np.concatenate(x_train_single_li, axis=0)
y_train_single = np.concatenate(y_train_single_li, axis=0)
x_val_single = np.concatenate(x_val_single_li, axis=0)
y_val_single = np.concatenate(y_val_single_li, axis=0)

print ('Single window of past history')
print (x_train_single.shape)
print (x_train_single[0])
print ('Single window of past history : {}'.format(x_train_single[0].shape))
print ('\n Target temperature to predict')
print (y_train_single.shape)
print (y_train_single[0])
print (x_val_single.shape)
print (y_val_single.shape)


## Recurrent neural network
BATCH_SIZE = 96
BUFFER_SIZE = 300

train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

single_step_model = tf.keras.models.Sequential()
single_step_model.add(tf.keras.layers.LSTM(12, input_shape=x_train_single.shape[-2:]))
single_step_model.add(tf.keras.layers.Dense(1))

single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

for x, y in val_data_single.take(1):
    print(single_step_model.predict(x).shape)

EVALUATION_INTERVAL = 200
EPOCHS = 30

single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS,
                                            steps_per_epoch=EVALUATION_INTERVAL,
                                            validation_data=val_data_single,
                                            validation_steps=50)

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

plot_train_history(single_step_history, 'Single Step Training and validation loss')

## Predict a single step future
# for x, y in val_data_single.take(3):
#     plot = show_plot([x[0][:, 1].numpy(), y[0].numpy(),
#                         single_step_model.predict(x)[0]], 12,
#                     'Single Step Prediction')
#     plot.show()