# https://www.tensorflow.org/tutorials/structured_data/time_series

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os, sys, re
import pandas as pd

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# mpl.rcParams['figure.figsize'] = (8, 6)
# mpl.rcParams['axes.grid'] = False

work_path = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\"
stack_csv = "stacked_data_200121.csv"
raw_csv = "stacked_data_200121_raw.csv"
features_li = ['JS', 'MM', 'Permits'] #prediction model features
target_feature = 'MM'
target_feature_window_size = 12 #rolling div window
model_file = 'multi_step_model_6_6.h5' #past_future


def init():
    ## The stacked dataset
    txt_path = work_path + stack_csv
    df = pd.read_csv(txt_path)

    ## The raw dataset
    raw_txt_path = work_path + raw_csv
    raw_df = pd.read_csv(raw_txt_path)

    ## Load prediction model
    model = tf.keras.models.load_model(model_file)
    model.summary()
    past_history = int(model_file.split('.')[0].split('_')[-2])
    future_target = int(model_file.split('.')[0].split('_')[-1])

    return df, raw_df, model, past_history, future_target

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

def build_multi_step_train_val_data(df, features_li, past_history):

    features = df[features_li]
    features.index = df['Date Time']
    features = features.dropna()

    dataset = features.values
    data_mean = dataset.mean(axis=0)
    data_std = dataset.std(axis=0)
    #dataset = (dataset-data_mean)/data_std
    
    # dataset[:, 1] -> y_train_multi
    x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 1],
                                                0, None, past_history, 0, 1)
                                                # future_target, STEP)                                            
    return x_val_multi, y_val_multi

def build_target_feature_raw_data(df, feature):
    feature_df = df[feature]
    feature_df.index = df['Date Time']
    feature_df = feature_df.dropna()                                       
    return feature_df

def create_time_steps(length):
    time_steps = []
    for i in range(-length, 0, 1):
        time_steps.append(i)
    return time_steps

def multi_step_pred_plot(history, prediction):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(prediction)

    plt.plot(num_in, np.array(history[:, 1]), label='History')
    #plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
    plt.plot(np.arange(num_out), np.array(prediction), 'ro',
            label='Predicted Future') #'bo', 'ro'
    plt.legend(loc='upper left')
    plt.show()

def extract_last_batch(x_val_multi, y_val_multi):
    val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
    val_data_num = len(y_val_multi)
    val_data_multi = val_data_multi.batch(1).repeat()

    final_x, final_y = None, None
    for x, y in val_data_multi.take(val_data_num):
        final_x, final_y = x, y
    return final_x, final_y

def extract_raw_last_batch(raw_df, region):
    region_raw_df = raw_df.loc[raw_df['Reg'] == region]
    region_raw_df = build_target_feature_raw_data(region_raw_df, target_feature)
    nrow = region_raw_df.shape[0]
    # target_feature_window_size = 12
    start_index = nrow-target_feature_window_size+1
    end_index = start_index + future_target
    raw_value_list = region_raw_df[start_index:end_index].tolist()
    return raw_value_list


## Init
df, raw_df, model, past_history, future_target = init()

## Build per region latest dataset
for region in list(df['Reg'].drop_duplicates()):
    region_df = df.loc[df['Reg'] == region]
    x_val_multi, y_val_multi = build_multi_step_train_val_data(region_df, features_li, past_history)

    ## extract last batch data
    final_x, final_y = extract_last_batch(x_val_multi, y_val_multi)
    
    ## Feed latest data to model, plot multi_step_prediction 
    prediction = model.predict(final_x)
    #print(' '.join(map(str, prediction[0])))
    print(region, prediction[0])
    #multi_step_pred_plot(final_x[0], prediction[0])

    ## extract raw data
    raw_value_list = extract_raw_last_batch(raw_df, region)

    index_list = list(map(lambda x, y: x*(y/100+1), raw_value_list, prediction[0]))
    print(region, index_list)
    total_change_rate = (index_list[-1]/index_list[0]-1)*100
    print(region, total_change_rate)

    ########################TO DO#############################
    ##########################################################