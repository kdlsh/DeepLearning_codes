from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os, sys, re
import pandas as pd
import scipy.stats as stats
from functools import reduce

os.chdir(os.path.dirname(os.path.realpath(__file__)))

##############################To do#######################################
# rate type data
# window size per data
##########################################################################

## parameter
Supply_Window_Size = 24 # month
Price_Window_Size = 12 # month

## supply int data type
path1 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\Permits_supply.txt"
path2 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\Starts_supply.txt"
path3 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\Completed_supply.txt"
path4 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\Unsold_supply.txt"
## index data type
path5 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\MM_index.txt"
path6 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\JS_index.txt"
path7 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\MG_index.txt"
## percent type
path8 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\JSratio_percent.txt"
## rate type
path9 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\JW_rate.txt"
path10 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\Interest_rate.txt"

path_li = [path1, path2, path3, path4, path5, path6, path7, path8, path9, path10]


def replace_reg_name(df):
    df.rename(columns={'지역(1)':'Reg', '지역':'Reg',
                       '구  분(1)':'Reg', '구분(1)':'Reg', 
                       '시도별(2)':'Reg'}, inplace=True)
    df['Reg'].replace({'전국':'Total', '소계':'Total', '계':'Total',
                    '수도권':'Cap', '수도권소계':'Cap',
                    '서울':'SO', '서울특별시':'SO',
                    '인천':'IC', '인천광역시':'IC',
                    '부산':'BS', '부산광역시':'BS',
                    '대구':'DG', '대구광역시':'DG',
                    '광주':'GJ', '광주광역시':'GJ',
                    '대전':'DJ', '대전광역시':'DJ',
                    '울산':'US', '울산광역시':'US',
                    '세종':'SJ', '세종특별자치시':'SJ',
                    '강원':'GW', '강원도':'GW', '경기':'GG', '경기도':'GG',
                    '충북':'CB', '충청북도':'CB', '충남':'CN', '충청남도':'CN',
                    '전북':'JB', '전라북도':'JB', '전남':'JN', '전라남도':'JN',
                    '경북':'GB', '경상북도':'GB', '경남':'GN', '경상남도':'GN',
                    '제주':'JJ', '제주도':'JJ', '제주특별자치도':'JJ', '제주특별자치시도':'JJ'
                    }, inplace =True)
    return df

def preprocess_df(path):
    df = pd.read_csv(path, sep='\t', dtype=str)
    df = replace_reg_name(df)
    df = df.set_index('Reg').T #transpose
    if df['SO'].str.contains(',').any():    
        df = df.applymap(lambda x: x.replace(',', ''))
    if df.isin(['-']).any().any():
        df.replace({'-': None}, inplace =True)
    df = df.drop(['SJ', 'Cap', 'Total'], axis=1, errors='ignore') #drop column
    df = df.dropna()
    df = df.apply(pd.to_numeric)
    return df

def supply_normalize(df):
    df = df.apply(lambda x : (x - x.mean())/x.mean())
    return df

def div_func(array):
    return ((array[-1]/array[0])-1)*100

def get_suffix_list(header_list):
    suffix_list = list(map(lambda x: '_'+x, header_list))
    return suffix_list

def build_merged_df(path_li, norm_flag, roll_flag):
    df_roll_list = []
    header_list = []
    for path in path_li:
        prefix, data_type = os.path.basename(path).split('.')[0].split('_')
        header_list.append(prefix)

        ## preprocessing
        df_pre = preprocess_df(path)
        if norm_flag and data_type == "supply":
            df_pre = supply_normalize(df_pre)

        ## window rolling
        if roll_flag:
            if data_type == "supply": # sum
                df_roll = df_pre.rolling(Supply_Window_Size).sum().dropna()
            elif data_type == "index": # apply div
                df_roll = df_pre.rolling(Price_Window_Size, center=True).apply(lambda x: div_func(x)).dropna()
                #df_roll = df_pre.rolling(Price_Window_Size).apply(lambda x: div_func(x)).dropna()
            elif data_type == "percent":
                df_roll = df_pre
            elif data_type == "rate":
                df_roll = df_pre
            else:
                sys.stderr.write('ERROR!! check raw data filename.\n')
                sys.exit
            df_roll_list.append(df_roll)
        else:
            df_roll_list.append(df_pre)

    ## add suffix
    suffix_list = get_suffix_list(header_list)
    for i in range(len(df_roll_list)):
        df_roll_list[i] = df_roll_list[i].add_suffix(suffix_list[i])
        df_roll_list[i]['Date Time'] = df_roll_list[i].index

    ## save merged dataframes
    df_final = reduce(lambda left,right: pd.merge(left,right,on='Date Time', how='outer'), df_roll_list)
    #df_final = reduce(lambda left,right: pd.merge(left,right,on='Date Time'), df_roll_list)
    df_final.index = df_final['Date Time']
    df_final = df_final.drop(columns=['Date Time'])

    df_final.to_csv("merged_data.csv", mode='w')
    return df_final, header_list

def build_stacked_df(df_merged, header_list):
    ## stack merged dataframe
    df_ = pd.DataFrame(index=['0', '1'], columns=header_list)
    df_ = df_.fillna(np.NaN)

    reg_list = list(map(lambda x:x.split('_')[0], list(df_merged.columns)))
    lookup = set()  # a temporary lookup set
    reg_list = [x for x in reg_list if x not in lookup and lookup.add(x) is None]

    suffix_list = get_suffix_list(header_list)
    for reg in reg_list:
        col_name_list = list(map(lambda x: reg+x, suffix_list))
        df_reg = df_merged[col_name_list]
        df_reg['Reg'] = reg

        rename_dic = dict(zip(col_name_list, header_list))
        df_ = pd.concat([df_, df_reg.rename(columns=rename_dic)])
    #df_ = df_.dropna()
    df_ = df_.drop(['0', '1'])
    df_ = df_[header_list+['Reg']]
    df_.index.name = 'Date Time'
    df_.to_csv("stacked_data.csv", mode='w')
    return df_


df_merged, header_list = build_merged_df(path_li, True, True) #norm_flag, roll_flag
#df_merged, header_list = build_merged_df(path_li, False, False) #norm_flag, roll_flag
df_stack = build_stacked_df(df_merged, header_list)



