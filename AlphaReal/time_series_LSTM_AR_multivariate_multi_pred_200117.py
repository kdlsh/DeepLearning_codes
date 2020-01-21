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

## The stacked dataset
txt_path = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\stacked_data_200121.csv"
df = pd.read_csv(txt_path)
features_li = ['JS', 'MM', 'Permits']

## The raw dataset
raw_txt_path = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\stacked_data_200121_raw.csv"
raw_df = pd.read_csv(raw_txt_path)
target_feature = 'MM'

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

def build_multi_step_train_val_data(df, features_li):

    features = df[features_li]
    features.index = df['Date Time']
    features = features.dropna()

    dataset = features.values
    data_mean = dataset.mean(axis=0)
    data_std = dataset.std(axis=0)
    #dataset = (dataset-data_mean)/data_std
    
    # dataset[:, 1] -> y_train_multi
    x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 1],
                                                0, None, past_history,
                                                future_target, STEP)                                            
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
    plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
            label='Predicted Future') #'bo', 'ro'
    plt.legend(loc='upper left')
    plt.show()


## Build per region latest dataset
for region in list(df['Reg'].drop_duplicates()):
    region_df = df.loc[df['Reg'] == region]
    x_val_multi, y_val_multi = build_multi_step_train_val_data(region_df, features_li)

    region_raw_df = raw_df.loc[raw_df['Reg'] == region]
    region_raw_df = build_target_feature_raw_data(region_raw_df, target_feature)
    nrow = region_raw_df.shape[0]
    target_feature_window_size = 12
    print(region_raw_df[(nrow-target_feature_window_size+1):nrow])
    sys.exit()
    #=B184*(D195/100+1)

# 2018. 09	108.4	8.835341365	
# 2018. 10	109.1	9.1	
# 2018. 11	109.1	8.234126984	
# 2018. 12	108.9	6.555772994	
# 2019. 01	108.5	4.72972973	
# 2019. 02	108.1	3.544061303	
# 2019. 03	107.7	2.767175573	
# 2019. 04	107.3	2.19047619	
# 2019. 05	107.1	1.80608365	
# 2019. 06	107	1.325757576	
# 2019. 07	107.1	0.563380282	
# 2019. 08	107.2	-1.10701107	
# 2019. 09	107.4	-1.558203483	
# 2019. 10	108.1	-0.916590284	
# 2019. 11	108.8	-0.091827365	
# 2019. 12	109.5988724	1.0127856	1.0127856
# 2020. 01	110.2345124	1.9745721	1.9745721
# 2020. 02	110.4600503	2.5627208	2.5627208
# 2020. 03	110.4919096	2.9747527	2.9747527
# 2020. 04	111.1034122	3.7380133	3.7380133
# 2020. 05	111.2181756	3.9422202	3.9422202



    val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
    val_data_num = len(y_val_multi)
    val_data_multi = val_data_multi.batch(1).repeat()

    final_x, final_y = None, None
    for x, y in val_data_multi.take(val_data_num):
        final_x, final_y = x, y
    
    ## Feed latest data to model, plot multi_step_prediction 
    print(region)
    prediction = model.predict(final_x)
    print(' '.join(map(str, prediction[0])))
    #multi_step_pred_plot(final_x[0], prediction[0])
    #sys.exit()

    ########################TO DO#############################
    # 1. parse to real change rate and price index
    ##########################################################
