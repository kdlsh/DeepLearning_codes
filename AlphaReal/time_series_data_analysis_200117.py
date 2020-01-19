from __future__ import absolute_import, division, print_function, unicode_literals
#import tensorflow as tf

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
    ## etc type; rolling div
    path11 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\Stock_div.txt"
    path12 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\Exchange_div.txt"
    ## 100-based index data type; rolling div
    path1 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\MG_div.txt"
    path2 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\MM_div.txt"
    path3 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\JS_div.txt"
    ## percent type
    path4 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\JSratio_percent.txt"
    ## supply int data type; normalize; rolling sum
    path5 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\Permits_sum.txt"
    path6 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\Starts_sum.txt"
    path7 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\Completed_sum.txt"
    ## supply int data type; normalize; rolling sub
    path8 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\Unsold_sub.txt"
    ## rate type; rolling sub
    path9 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\Interest_sub.txt"
    path10 = "D:\\workspace\\DeepLearning_codes\\AlphaReal\\raw_data\\JW_sub.txt"

    norm_flag_list = [False, False, False, False, False, False, True, True, True, True, False, False]
    path_li = [path11, path12, path1, path2, path3, path4, path5, path6, path7, path8, path9, path10]
    window_li = [1, 1, 12, 12, 12, 1, 32, 32, 24, 12, 6, 6] # month

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

def preprocess_df(path, drop_reg_li):
    df = pd.read_csv(path, sep='\t', dtype=str)
    df = replace_reg_name(df)
    df = df.set_index('Reg').T #transpose
    df = df.astype(str)
    if df['SO'].str.contains(',').any():    
        df = df.applymap(lambda x: x.replace(',', ''))
    if df.isin(['-','nan']).any().any():
        df.replace({'-':None, 'nan':None}, inplace =True)
    #df = df.drop(['SJ', 'Cap', 'Total'], axis=1, errors='ignore') #drop column
    df = df.drop(drop_reg_li, axis=1, errors='ignore') #drop column
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

def build_merged_df(path_li, window_li, norm_flag_li, drop_reg_li, save_flag):
    df_roll_list = []
    header_list = []
    for path, window, norm_flag in zip(path_li, window_li, norm_flag_li):
        prefix, data_type = os.path.basename(path).split('.')[0].split('_')
        header_list.append(prefix)

        ## preprocessing
        df_pre = preprocess_df(path, drop_reg_li)
        if norm_flag:
            df_pre = supply_normalize(df_pre)

        ## window rolling
        if window > 1:
            if data_type == "sum": # cumulative sum; supply type
                df_roll = df_pre.rolling(window).sum().dropna()
            elif data_type == "div": # two point div; index type
                #df_roll = df_pre.rolling(window, center=True).apply(lambda x: div_func(x)).dropna()
                df_roll = df_pre.rolling(window).apply(lambda x: div_func(x)).dropna()
            elif data_type == "percent":
                df_roll = df_pre
            elif data_type == "sub": # two point sub; rate type
                #df_roll = df_pre.rolling(window, center=True).apply(lambda x: sub_func(x)).dropna()
                df_roll = df_pre.rolling(window).apply(lambda x: sub_func(x)).dropna()
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

def pair_data_config(path_li, norm_flag_li, cat1, win1, cat2, win2):
    cat1_indice = [i for i, s in enumerate(path_li) if cat1 in s][0]
    cat2_indice = [i for i, s in enumerate(path_li) if cat2 in s][0]
    path_li = [path_li[cat1_indice], path_li[cat2_indice]]
    window_li = [win1, win2]
    norm_flag_li = [norm_flag_li[cat1_indice], norm_flag_li[cat2_indice]]
    return path_li, window_li, norm_flag_li

def multi_data_config(path_li, norm_flag_li, cat_list, win_list):
    cat_indice_list = []
    for cat in cat_list:
        for i, s in enumerate(path_li):
            if (cat+'_') in s:
                cat_indice_list.append(i)
    path_li = [path_li[i] for i in cat_indice_list]
    window_li = win_list
    norm_flag_li = [norm_flag_li[i] for i in cat_indice_list]
    return path_li, window_li, norm_flag_li

def build_merged_and_stacked_total_table(drop_reg_li):
    path_li, window_li, norm_flag_li = config()
    df_merged, header_list = build_merged_df(path_li, window_li, norm_flag_li, drop_reg_li, True)
    #df_stack = build_stacked_df(df_merged, header_list, True)
    build_stacked_df(df_merged, header_list, True)

def build_pair_corr_table(cat1, start1, end1, cat2, start2, end2, corr_cutoff, drop_reg_li):
    path_li, window_li, norm_flag_li = config()

    p_corr_li = []
    for win1 in range(start1, end1):
        for win2 in range(start2, end2):
            path_li, window_li, norm_flag_li = pair_data_config(path_li, norm_flag_li, cat1, win1, cat2, win2)
            df_merged, header_list = build_merged_df(path_li, window_li, norm_flag_li, drop_reg_li, False)
            df_stack = build_stacked_df(df_merged, header_list, False)
            pearson_r = df_stack.corr().iloc[0,1]
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

def plot_pair_pearson_corr(df, cat1, cat2, filename_prefix):
    df = df.dropna()
    overall_pearson_r = df.corr().iloc[0,1]    

    ## Rolling correlation
    ROLLING_WINDOW_SIZE = 6
    r_window_size = 12 #36 for Total
    # Interpolate missing data.
    df_interpolated = df.interpolate()
    # Compute rolling window synchrony
    rolling_r = df_interpolated[cat1].rolling(window=r_window_size, center=True).corr(df_interpolated[cat2])
    f,ax=plt.subplots(2,1,figsize=(14,6),sharex=True)
    df.rolling(window=ROLLING_WINDOW_SIZE,center=True).median().plot(ax=ax[0])
    ax[0].set(xlabel='Date',ylabel='Rate')
    ax[0].axhline(0, color='r', linewidth=1)
    rolling_r.plot(ax=ax[1])
    ax[1].set(xlabel='Date',ylabel='Pearson r')
    overall_corr = f"(Overall Pearson r = {np.round(overall_pearson_r,2)})"
    #plt.suptitle("Rate data and rolling window correlation " +overall_corr)
    plt.suptitle(filename_prefix+' '+overall_corr)
    f.savefig(filename_prefix+'.png')

def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation. 
    Shifted data filled with NaNs 
    
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else: 
        return datax.corr(datay.shift(lag))

def plot_pair_lagged_corr(df, cat1, cat2, filename_prefix):
    df = df.dropna()
    d1 = df[cat1]
    d2 = df[cat2]
    seconds = 6
    fps = 10
    rs = [crosscorr(d1,d2, lag) for lag in range(-int(seconds*fps-1),int(seconds*fps))]
    offset = np.ceil(len(rs)/2)-np.argmax(np.abs(rs))
    f,ax=plt.subplots(figsize=(14,4))
    ax.plot(rs)
    ax.axvline(np.ceil(len(rs)/2),color='k',linestyle='--',label='Center')
    ax.axvline(np.argmax(np.abs(rs)),color='r',linestyle='--',label='Peak synchrony')
    ax.axhline(0, color='r', linewidth=1)
    ax.set(title=f'Offset = {offset} Dates\nS1 leads <> S2 leads',ylim=[-.81,.81],xlim=[0,seconds*fps*2], xlabel='Offset',ylabel='Pearson r')
    ax.set_xticklabels([int(item-seconds*fps) for item in ax.get_xticks()])
    plt.legend()
    f.savefig(filename_prefix+'_lagged.png')
    print(filename_prefix, offset)

def plot_pair_corr(df, cat1, cat2, filename_prefix):
    df = df.dropna()
    overall_pearson_r = df.corr().iloc[0,1]    

    ## Rolling correlation
    ROLLING_WINDOW_SIZE = 6
    r_window_size = 12 #36 for Total
    df_interpolated = df.interpolate()
    # Compute rolling window synchrony
    rolling_r = df_interpolated[cat1].rolling(window=r_window_size, center=True).corr(df_interpolated[cat2])
    f,ax=plt.subplots(3,1,figsize=(14,12),sharex=False)
    df.rolling(window=ROLLING_WINDOW_SIZE,center=True).median().plot(ax=ax[0])
    ax[0].set(xlabel='Date',ylabel='Rate')
    ax[0].axhline(0, color='r', linewidth=1)
    rolling_r.plot(ax=ax[1])
    ax[1].set(xlabel='Date',ylabel='Pearson r')
    overall_corr = f"(Overall Pearson r = {np.round(overall_pearson_r,2)})"
    plt.suptitle(filename_prefix+' '+overall_corr)
    
    ## Time lagged cross correlation
    d1 = df[cat1]
    d2 = df[cat2]
    seconds = 6
    fps = 10
    rs = [crosscorr(d1,d2, lag) for lag in range(-int(seconds*fps-1),int(seconds*fps))]
    offset = np.ceil(len(rs)/2)-np.argmax(np.abs(rs))
    
    ax[2].plot(rs)
    ax[2].axvline(np.ceil(len(rs)/2),color='k',linestyle='--',label='Center')
    ax[2].axvline(np.argmax(np.abs(rs)),color='r',linestyle='--',label='Peak synchrony')
    ax[2].axhline(0, color='r', linewidth=1)
    ax[2].set(title=f'Offset = {offset} Dates',ylim=[-.81,.81],xlim=[0,seconds*fps*2], xlabel='Offset',ylabel='Pearson r')
    ax[2].set_xticklabels([int(item-seconds*fps) for item in ax[2].get_xticks()])
    plt.legend()
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    f.savefig(filename_prefix+'.png')
    print(filename_prefix, offset)

def plot_total_pair_corr(cat1, win1, cat2, win2, drop_reg_li):
    path_li, window_li, norm_flag_li = config()
    path_li, window_li, norm_flag_li = pair_data_config(path_li, norm_flag_li, cat1, win1, cat2, win2)
    df_merged, header_list = build_merged_df(path_li, window_li, norm_flag_li, drop_reg_li, False)
    df_stack = build_stacked_df(df_merged, header_list, False)
    filename_prefix = '_'.join([cat1, str(win1), cat2, str(win2)])
    # plot_pair_pearson_corr(df_stack, cat1, cat2, filename_prefix)
    # plot_pair_lagged_corr(df_stack, cat1, cat2, filename_prefix)
    plot_pair_corr(df_stack, cat1, cat2, filename_prefix)

def plot_total_multi_corr(cat_list, win_list, drop_reg_li):
    path_li, window_li, norm_flag_li = config()
    path_li, window_li, norm_flag_li = multi_data_config(path_li, norm_flag_li, cat_list, win_list)
    df_merged, header_list = build_merged_df(path_li, window_li, norm_flag_li, drop_reg_li, False)
    df_stack = build_stacked_df(df_merged, header_list, False)
    filename_prefix = '_'.join(cat_list + list(map(str, win_list)))
    
    df = df_stack.dropna()
    overall_pearson_r = df.corr() 
    f,ax=plt.subplots(figsize=(14,5))
    df.rolling(window=3,center=True).median().plot(ax=ax)
    ax.set(xlabel='Date',ylabel='Rate',title=f"Overall Pearson r = {np.round(overall_pearson_r,2)}")
    f.subplots_adjust(top=0.80)
    f.savefig(filename_prefix+'.png')

def plot_per_reg_pair_corr(cat1, win1, cat2, win2, drop_reg_li):
    path_li, window_li, norm_flag_li = config()
    path_li, window_li, norm_flag_li = pair_data_config(path_li, norm_flag_li, cat1, win1, cat2, win2)
    df_merged, header_list = build_merged_df(path_li, window_li, norm_flag_li, drop_reg_li, False)
    df_stack = build_stacked_df(df_merged, header_list, False)

    for region in list(df_stack['Reg'].drop_duplicates()):
        region_df = df_stack.loc[df_stack['Reg'] == region]
        filename_prefix = '_'.join([cat1, str(win1), cat2, str(win2),region])
        # plot_pair_pearson_corr(region_df, cat1, cat2, filename_prefix)
        # plot_pair_lagged_corr(region_df, cat1, cat2, filename_prefix)
        plot_pair_corr(region_df, cat1, cat2, filename_prefix)

def plot_per_reg_multi_corr(cat_list, win_list, drop_reg_li):
    path_li, window_li, norm_flag_li = config()
    path_li, window_li, norm_flag_li = multi_data_config(path_li, norm_flag_li, cat_list, win_list)
    df_merged, header_list = build_merged_df(path_li, window_li, norm_flag_li, drop_reg_li, False)
    df_stack = build_stacked_df(df_merged, header_list, False)

    for region in list(df_stack['Reg'].drop_duplicates()):
        region_df = df_stack.loc[df_stack['Reg'] == region]
        filename_prefix = '_'.join(cat_list + list(map(str, win_list))+[region])
    
        df = region_df.dropna()
        overall_pearson_r = df.corr()
        f,ax=plt.subplots(figsize=(14,5))
        df.rolling(window=3,center=True).median().plot(ax=ax)
        ax.set(xlabel='Date',ylabel='Rate',title=f"Overall Pearson r = {np.round(overall_pearson_r,2)}")
        ax.axhline(0, color='r', linewidth=1)
        f.subplots_adjust(top=0.80)
        f.savefig(filename_prefix+'.png')

def plot_multi_reg(cat, win, reg_list, drop_reg_li):
    path_li, window_li, norm_flag_li = config()
    path_li, window_li, norm_flag_li = multi_data_config(path_li, norm_flag_li, [cat], [win])
    df_merged, header_list = build_merged_df(path_li, window_li, norm_flag_li, drop_reg_li, False)

    df_merged.columns = list(map(lambda x:x.replace('_'+cat,''), df_merged.columns))
    df = df_merged[reg_list]
    df = df.dropna()
    filename_prefix = '_'.join([cat, str(win)] + reg_list)

    overall_pearson_r = df.corr()
    f,ax=plt.subplots(figsize=(14,5))
    df.rolling(window=3,center=True).median().plot(ax=ax)
    ax.set(xlabel='Date',ylabel='Rate',title=f"Overall Pearson r = {np.round(overall_pearson_r,2)}")
    ax.axhline(0, color='r', linewidth=1)
    f.subplots_adjust(top=0.80)
    f.savefig(filename_prefix+'.png')



if __name__ == "__main__":
    ## Categories
    total_cat = ['Stock','Exchange', 'MG','MM','JS','JSratio','Permits','Starts','Completed',
                'Unsold','Interest','JW']

    ## Build total table; input stacked data for LSTM learning
    # build_merged_and_stacked_total_table(['SJ','Total','Cap'])

    ## Grid search (Build two category pearson correlation table)
    # build_pair_corr_table('MM', 1, 24, 'JS', 1, 24, 0.6, ['SJ','Total','Cap']) #[1, 1, 0.8871]
    # build_pair_corr_table('MM', 1, 24, 'MG', 1, 24, 0.4, ['SJ','Total','Cap']) #[1, 1, 0.8622]
    # build_pair_corr_table('MM', 1, 48, 'Unsold', 1, 42, 0.4, ['SJ','Total','Cap']) #[41, 40, -0.5465]
    # build_pair_corr_table('MM', 1, 48, 'Permits', 1, 42, 0.4, ['SJ','Total','Cap']) #[23, 38, -0.5004]
    # build_pair_corr_table('MM', 1, 48, 'Starts', 1, 42, 0.4, ['SJ','Total','Cap']) #[23, 41, -0.5468]
    # build_pair_corr_table('MM', 1, 24, 'Completed', 1, 42, 0.4, ['SJ','Total','Cap']) #[18, 14, -0.5372]
    # build_pair_corr_table('JS', 1, 24, 'Completed', 1, 42, 0.4, ['SJ','Total','Cap']) #[18, 15, -0.6938]
    # build_pair_corr_table('MM', 1, 36, 'Interest', 1, 24, 0.2, ['SJ','Total','Cap']) #[1, 1, -0.6553]
    # build_pair_corr_table('MM', 1, 24, 'Stock', 1, 24, 0.3, ['SJ','Total','Cap']) #[1, 1, 0.7664]

    ## Plot correlation
    # plot_total_pair_corr('MM', 3, 'JS', 3, ['SJ','Total','Cap'])
    # plot_per_reg_pair_corr('MM', 3, 'JS', 3, ['SJ','Total','Cap'])
    # plot_total_pair_corr('MM', 3, 'Unsold', 1, ['SJ','Total','Cap'])
    # plot_per_reg_pair_corr('MM', 3, 'Unsold', 1, ['SJ','Total','Cap'])
    # plot_total_pair_corr('MM', 12, 'Permits', 24, ['SJ','Total','Cap'])
    # plot_per_reg_pair_corr('MM', 12, 'Permits', 24, ['SJ','Total','Cap'])
    # plot_per_reg_pair_corr('MM', 12, 'Interest', 12, ['SJ','Total','Cap'])
    # plot_per_reg_pair_corr('JS', 12, 'Completed', 12, ['SJ','Total','Cap'])

    ## Multi category plot
    # plot_total_multi_corr(['Permits','Starts','Completed'], [12,12,12], ['SJ','Total','Cap'])
    # plot_per_reg_multi_corr(['Permits','Starts','Completed'], [12,12,12], ['SJ','Total','Cap'])
    # plot_per_reg_multi_corr(['MM','MG','Stock'], [12,12,12], ['SJ','Total','Cap'])

    ## Multi region plot
    # plot_multi_reg('Permits', 32, ['BS','GN','GG','SO'], ['SJ','Total','Cap'])
    # plot_multi_reg('MM', 12, ['BS','GN','GG','SO'], ['SJ','Total','Cap'])
    # plot_multi_reg('JS', 6, ['BS','GN','GB','SO'], ['SJ','Total','Cap'])
    # plot_multi_reg('Completed', 6, ['BS','GN','GB','SO'], ['SJ','Total','Cap'])

    # plot_multi_reg('Permits', 6, ['SJ','DJ','CB','CN'], ['Total','Cap'])
    # plot_multi_reg('MM', 6, ['SJ','DJ','CB','CN'], ['Total','Cap'])
    # plot_multi_reg('JS', 6, ['SJ','DJ','CB','CN'], ['Total','Cap'])
    # plot_multi_reg('Completed', 6, ['SJ','DJ','CB','CN'], ['Total','Cap'])

    # plot_multi_reg('Permits', 6, ['GJ','JN','JB'], ['SJ','Total','Cap'])
    # plot_multi_reg('MM', 6, ['GJ','JN','JB'], ['SJ','Total','Cap'])
    # plot_multi_reg('JS', 6, ['GJ','JN','JB'], ['SJ','Total','Cap'])
    # plot_multi_reg('Completed', 6, ['GJ','JN','JB'], ['SJ','Total','Cap'])

    # plot_multi_reg('Permits', 32, ['SO','IC','GG'], ['SJ','Total'])
    # plot_multi_reg('MM', 12, ['SO','IC','GG','Cap'], ['SJ','Total'])
    # plot_multi_reg('JS', 6, ['SO','IC','GG','Cap'], ['SJ','Total'])
    # plot_multi_reg('Completed', 6, ['SO','IC','GG','Cap'], ['SJ','Total'])
    
