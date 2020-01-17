# https://www.tensorflow.org/tutorials/structured_data/time_series

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os, sys, re
import pandas as pd

os.chdir(os.path.dirname(os.path.realpath(__file__)))

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


## Parameters
past_history = 6  #24
future_target = 0
STEP = 1

## The dataset
txt_path = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\stacked_data.csv"
df = pd.read_csv(txt_path)
print(df.head())
df = df.drop(['Unsold','Completed','Starts','JSratio','JW'], axis=1)
df = df.dropna()
print(df.head())


########################TO DO#############################
# 2. build per region latest dataset
# 3. feed latest data to prediction model
# 4. plot multi_step_latest_plot
# 5. 3 or 5 category
##########################################################

## Load prediction model
model = tf.keras.models.load_model('multi_step_model_6_6.h5')
model.summary()


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size +1

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)

def build_multi_step_train_val_data(df):

    features_considered = ['JS', 'MM', 'Permits']
    features = df[features_considered]
    features.index = df['Date Time']

    dataset = features.values
    data_mean = dataset.mean(axis=0)
    data_std = dataset.std(axis=0)
    #dataset = (dataset-data_mean)/data_std
    
    # dataset[:, 1] -> y_train_multi
    x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 1],
                                                0, None, past_history,
                                                future_target, STEP)                                            
    return x_val_multi, y_val_multi


for region in list(df['Reg'].drop_duplicates()):
    region_df = df.loc[df['Reg'] == region]
    x_val_multi, y_val_multi = build_multi_step_train_val_data(region_df)

    val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
    val_data_num = len(y_val_multi)
    val_data_multi = val_data_multi.batch(1).repeat()

    final_x, final_y = None, None
    for x, y in val_data_multi.take(val_data_num):
        final_x, final_y = x, y
    
    print(region)
    print(final_x)
    print(final_y)
    print(model.predict(final_x).shape)
    sys.exit()