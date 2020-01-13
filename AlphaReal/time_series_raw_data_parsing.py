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

def config():
    ## path order -> long 'Date Time'
    ## 100-based index data type; rolling div
    path1 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\MG_index.txt"
    path2 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\MM_index.txt"
    path3 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\JS_index.txt"
    ## percent type
    path4 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\JSratio_percent.txt"
    ## supply int data type; normalize; rolling sum
    path5 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\Permits_supply.txt"
    path6 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\Starts_supply.txt"
    path7 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\Completed_supply.txt"
    path8 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\Unsold_supply.txt"
    ## rate type; rolling sub
    path9 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\Interest_rate.txt"
    path10 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\JW_rate.txt"
    ## etc type; rolling div
    path11 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\Stock_index.txt"
    path12 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\Exchange_index.txt"

    norm_flag_list = [False, False, False, False, True, True, True, True, False, False, False, False]
    path_li = [path1, path2, path3, path4, path5, path6, path7, path8, path9, path10, path11, path12]
    window_li = [12, 12, 12, 1, 32, 32, 24, 12, 6, 6, 1, 1] # month

    return path_li, window_li, norm_flag_list

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
    df = df.astype(str)
    if df['SO'].str.contains(',').any():    
        df = df.applymap(lambda x: x.replace(',', ''))
    if df.isin(['-','nan']).any().any():
        df.replace({'-':None, 'nan':None}, inplace =True)
    df = df.drop(['SJ', 'Cap', 'Total'], axis=1, errors='ignore') #drop column
    df = df.dropna()
    df = df.apply(pd.to_numeric)
    return df

def supply_normalize(df):
    df = df.apply(lambda x : (x - x.mean())/x.mean())
    return df

def div_func(array):
    return ((array[-1]/array[0])-1)*100

def sub_func(array):
    return (array[-1] - array[0])

def get_suffix_list(header_list):
    suffix_list = list(map(lambda x: '_'+x, header_list))
    return suffix_list

def build_merged_df(path_li, window_li, norm_flag_li, save_flag):
    df_roll_list = []
    header_list = []
    for path, window, norm_flag in zip(path_li, window_li, norm_flag_li):
        prefix, data_type = os.path.basename(path).split('.')[0].split('_')
        header_list.append(prefix)

        ## preprocessing
        df_pre = preprocess_df(path)
        if norm_flag and data_type == "supply":
            df_pre = supply_normalize(df_pre)

        ## window rolling
        if window > 1:
            if data_type == "supply": # sum
                df_roll = df_pre.rolling(window).sum().dropna()
            elif data_type == "index": # apply div
                df_roll = df_pre.rolling(window, center=True).apply(lambda x: div_func(x)).dropna()
                #df_roll = df_pre.rolling(window).apply(lambda x: div_func(x)).dropna()
            elif data_type == "percent":
                df_roll = df_pre
            elif data_type == "rate":
                df_roll = df_pre.rolling(window, center=True).apply(lambda x: sub_func(x)).dropna()
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

    if save_flag:
        df_final.to_csv("merged_data.csv", mode='w')

    return df_final, header_list

def build_stacked_df(df_merged, header_list, save_flag):
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
    if save_flag:
        df_.to_csv("stacked_data.csv", mode='w')
    return df_

def pair_data_config(path_li, window_li, norm_flag_li, cat1, win1, cat2, win2):
    cat1_indice = [i for i, s in enumerate(path_li) if cat1 in s][0]
    cat2_indice = [i for i, s in enumerate(path_li) if cat2 in s][0]
    path_li = [path_li[cat1_indice], path_li[cat2_indice]]
    window_li = [win1, win2]
    norm_flag_li = [norm_flag_li[cat1_indice], norm_flag_li[cat2_indice]]
    return path_li, window_li, norm_flag_li

def build_merged_and_stacked_total_table():
    path_li, window_li, norm_flag_li = config()
    df_merged, header_list = build_merged_df(path_li, window_li, norm_flag_li, True)
    #df_stack = build_stacked_df(df_merged, header_list, True)
    build_stacked_df(df_merged, header_list, True)

def get_p_corr(df):
    df = df.dropna()
    overall_pearson_r = df.corr().iloc[0,1]
    return overall_pearson_r

def build_pair_corr_table(cat1, start1, end1, cat2, start2, end2, corr_cutoff):
    path_li, window_li, norm_flag_li = config()

    p_corr_li = []
    for win1 in range(start1, end1):
        for win2 in range(start2, end2):
            path_li, window_li, norm_flag_li = pair_data_config(path_li, window_li, norm_flag_li, cat1, win1, cat2, win2)
            df_merged, header_list = build_merged_df(path_li, window_li, norm_flag_li, False)
            df_stack = build_stacked_df(df_merged, header_list, False)
            pearson_r = get_p_corr(df_stack)
            if abs(pearson_r) > corr_cutoff:
                p_corr_li.append([win1, win2, pearson_r])

    wfilename = '_'.join([cat1, str(end1), cat2, str(end2)]) + '.csv'
    wfile = open(wfilename, 'w')    
    wfile.write(','.join([cat1, cat2, 'p_corr']) + '\n')    
    for li in p_corr_li:
        wfile.write(','.join(map(str, li)) + '\n')
    wfile.close()

    p_corr_li_sorted = sorted(p_corr_li, key=lambda x:abs(x[-1]), reverse=True)
    print(p_corr_li_sorted[0])



if __name__ == "__main__":
    ## Categories
    total_cat = ['MG','MM','JS','JSratio','Permits','Starts','Completed',
                'Unsold','Interest','JW','Stock','Exchange']

    ## Build total table; input stacked data for LSTM learning
    #build_merged_and_stacked_total_table()

    ## Build two category pearson correlation table
    build_pair_corr_table('MM', 1, 24, 'JS', 1, 24, 0.6) #[1, 1, 0.8871]
    build_pair_corr_table('MM', 1, 24, 'MG', 1, 24, 0.4) #[1, 1, 0.8622]
    build_pair_corr_table('MM', 1, 48, 'Unsold', 1, 42, 0.4) #[1, 29, -0.7003]
    build_pair_corr_table('MM', 1, 48, 'Permits', 1, 42, 0.4) #[23, 38, -0.5004]
    build_pair_corr_table('MM', 1, 48, 'Starts', 1, 42, 0.4) #[23, 41, -0.5468]
    build_pair_corr_table('MM', 1, 24, 'Completed', 1, 42, 0.4) #[18, 14, -0.5372]
    build_pair_corr_table('JS', 1, 24, 'Completed', 1, 42, 0.4) #[18, 15, -0.6938]
    build_pair_corr_table('MM', 1, 36, 'Interest', 1, 24, 0.2) #[1, 1, -0.6553]
    build_pair_corr_table('MM', 1, 24, 'Stock', 1, 24, 0.3) #[1, 1, 0.7664]