from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os, sys, re
import pandas as pd
import scipy.stats as stats

os.chdir(os.path.dirname(os.path.realpath(__file__)))


## parameter
#P_window_size = 12 # month
P_start, P_end = 1, 60
#M_window_size = 3 # month
M_start, M_end = 1, 36

## int data type
p_path1 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\Permits_raw.txt"
p_path2 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\Starts_raw.txt"
p_path3 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\Completed_raw.txt"
p_path_li = [p_path1, p_path2, p_path3]
## index data type
m_path1 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\MM_raw.txt"
m_path2 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\JS_raw.txt"
m_path_li = [m_path1, m_path2]
## corr cutoff
corr_cutoff = 0.4


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

def get_corr_merge_stack_df(P_window_size, M_window_size, p_path, m_path):
    ## window rolling sum
    df_p = preprocess_df(p_path)
    df_permit = df_p.rolling(P_window_size).sum().dropna()

    ## window rolling apply div
    df_m = preprocess_df(m_path)   
    #df_MM = df_m.rolling(M_window_size, center=True).apply(lambda x: div_func(x)).dropna()
    df_MM = df_m.rolling(M_window_size).apply(lambda x: div_func(x)).dropna()

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
    if abs(overall_pearson_r) > corr_cutoff:
        # print(P_window_size, M_window_size)
        # print(f"Pandas computed Pearson r: {overall_pearson_r}")
        return [P_window_size, M_window_size, overall_pearson_r]

def write_corr_table(p_path, m_path):
    p_corr_li_total = []
    for p in range(P_start, P_end): # month
        for m in range(M_start, M_end): # month
            p_corr_li = get_corr_merge_stack_df(p, m, p_path, m_path)
            if p_corr_li != None:
                p_corr_li_total.append(p_corr_li)

    ## write result file
    p_prefix = os.path.basename(p_path).replace('_raw.txt', '')
    m_prefix = os.path.basename(m_path).replace('_raw.txt', '')

    wfile = open(p_prefix+"_"+m_prefix+".tab", 'w')    
    wfile.write('\t'.join([p_prefix, m_prefix, 'p_corr']) + '\n')    
    for li in p_corr_li_total:
        wfile.write('\t'.join(map(str, li)) + '\n')
    wfile.close()

    #p_corr_li_sorted = sorted(p_corr_li_total, key=lambda x:x[-1])
    #print(p_corr_li_sorted[0])


if __name__ == "__main__":

    for p_path in p_path_li:
        for m_path in m_path_li:    
            write_corr_table(p_path, m_path)




