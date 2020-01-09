from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os, sys, re
import pandas as pd

import seaborn as sns
import scipy.stats as stats
#%matplotlib inline
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


## parameter
#P_window_size = 12 # month
#M_window_size = 3 # month
## int data type
#p_path = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\Permits_raw.txt"
p_path = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\Starts_raw.txt"
#p_path = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\Completed_raw.txt"
## index data type
m_path = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\MM_raw.txt"
#m_path = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\JS_raw.txt"


def preprocess_df(path):
    ## The permits dataset
    txt_path = path
    df = pd.read_csv(txt_path, sep='\t')

    ## Preprocessing
    df = df.set_index('Reg').T #transpose
    df = df.drop(['SJ', 'Cap', 'Total'], axis=1, errors='ignore') #drop column
    if df.isin([',']).any().any():
        df = df.applymap(lambda x: x.replace(',', ''))
    if df.isin(['-']).any().any():
        df.replace({'-': None}, inplace =True)
    df = df.dropna()
    df = df.apply(pd.to_numeric)
    df.describe()
    return df

def build_MM_df(path):
    ## The MM dataset
    txt_path = path
    df = pd.read_csv(txt_path, sep='\t')

    ## Preprocessing
    df = df.set_index('Reg').T #transpose
    df = df.drop(['SJ', 'Cap', 'Total'], axis=1, errors='ignore') #drop column
    df = df.apply(pd.to_numeric)
    df.describe()
    return df

def div_func(array):
    return ((array[-1]/array[0])-1)*100

def get_corr_merge_stack_df(P_window_size, M_window_size):
    ## window rolling sum
    #df_p = build_permit_df(p_path)
    df_p = preprocess_df(p_path)
    df_permit = df_p.rolling(P_window_size).sum().dropna()

    ## window rolling apply div
    #df_m = build_MM_df(m_path)
    df_m = preprocess_df(m_path)   
    df_MM = df_m.rolling(M_window_size, center=True).apply(lambda x: div_func(x)).dropna()

    ## merge dataframe
    df_MM['Date Time'] = df_MM.index
    df_permit['Date Time'] = df_permit.index
    df_merged = pd.merge(df_MM, df_permit, on='Date Time', suffixes=('_A','_B'))
    df_merged.index = df_merged['Date Time']
    #return df_merged

    ## stack merged dataframe
    df_ = pd.DataFrame(index=['0', '1'], columns=['A', 'B'])
    df_ = df_.fillna(np.NaN)
    df_

    for reg in list(df_p.columns):
        reg_A = reg+"_A"
        reg_B = reg+"_B"   
        merged_reg = pd.merge(df_merged[reg_A], df_merged[reg_B], left_index=True, right_index=True)    
        df_ = pd.concat([df_,merged_reg.rename(columns={reg_A:'A', reg_B:'B'})])
    df_merged_stack = df_.dropna()

    ## Pearson correlation
    df = df_merged_stack[['A','B']]

    overall_pearson_r = df.corr().iloc[0,1]
    if abs(overall_pearson_r) > 0.05:
        print(P_window_size, M_window_size)
        print(f"Pandas computed Pearson r: {overall_pearson_r}")

    #r, p = stats.pearsonr(df.dropna()['Permits'], df.dropna()['MM'])
    #print(f"Scipy computed Pearson r: {r} and p-value: {p}")


for p in range(1,36): # month 6~36
    for m in range(1,12): # month 1~12
        get_corr_merge_stack_df(p, m)