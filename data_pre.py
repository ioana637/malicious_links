# 1. Load original dataset
# 2. Clear NaN or None values from the dataset
# 3. Categorical features in two formats
# 4. Normalize all data with MinMax Scaler and StandardScaler
# 5. Randomly split data into training and testing
import math
from datetime import *
from math import log

import numpy as np
import pandas as pd


# 1. Load original dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils import convert_string_to_datetime, convert_datetime_to_timestamp
def load_dataset(file = 'data/dataset_original.csv'):
    df = pd.read_csv(file)
    return df
# print(df.to_string())

# 2. Clear NaN or None values from the dataset
def clear_dataset(df):
    # 2.1 Clear dataset from - or -- values, replace them with the value e.g. None for categorical columns
    df['CHARSET'] = df['CHARSET'].apply(lambda x: x.lower() if(x != 'None') else x)
    # df["SERVER"] = df['SERVER'].replace('nan', 'None')
    # print(df['SERVER'].unique())
    df["WHOIS_COUNTRY"] = df['WHOIS_COUNTRY'].replace('United Kingdom', 'GB')
    df["WHOIS_COUNTRY"] = df['WHOIS_COUNTRY'].replace('UK', 'GB')
    df["WHOIS_COUNTRY"] = df['WHOIS_COUNTRY'].replace("[u'GB'; u'UK']", 'GB')
    df["WHOIS_COUNTRY"] = df['WHOIS_COUNTRY'].replace('ru', 'RU')
    df["WHOIS_COUNTRY"] = df['WHOIS_COUNTRY'].replace('se', 'SE')
    df["WHOIS_COUNTRY"] = df['WHOIS_COUNTRY'].replace('us', 'US')
    df["WHOIS_COUNTRY"] = df['WHOIS_COUNTRY'].replace('Cyprus', 'CY')
    # print(df['WHOIS_COUNTRY'].unique())
    df["WHOIS_STATEPRO"] = df['WHOIS_STATEPRO'].replace('NONE', 'None')
    df["WHOIS_STATEPRO"] = df['WHOIS_STATEPRO'].replace('-', 'None')
    df["WHOIS_STATEPRO"] = df['WHOIS_STATEPRO'].replace('--', 'None')
    df["WHOIS_STATEPRO"] = df['WHOIS_STATEPRO'].replace('Other', 'None')
    df["WHOIS_STATEPRO"] = df['WHOIS_STATEPRO'].replace('NOT APPLICABLE', 'None')
    df["WHOIS_STATEPRO"] = df['WHOIS_STATEPRO'].replace('Not Applicable', 'None')
    df['WHOIS_STATEPRO'] = df['WHOIS_STATEPRO'].apply(lambda x: x.upper() if(x != 'None') else x)
    # print(df['WHOIS_STATEPRO'].unique())

    # 2.2 Clear DATE columns
    df.loc[df["WHOIS_REGDATE"] == "None", "WHOIS_REGDATE"] = "01/01/1970 00:00"
    df.loc[df["WHOIS_REGDATE"] == "b", "WHOIS_REGDATE"] = "01/01/1970 00:00"
    df.loc[df["WHOIS_REGDATE"] == "0", "WHOIS_REGDATE"] = "01/01/1970 00:00"
    df['WHOIS_REGDATE'] = df['WHOIS_REGDATE'].apply(lambda x: convert_string_to_datetime(x))

    df.loc[df["WHOIS_UPDATED_DATE"] == "None", "WHOIS_UPDATED_DATE"] = "01/01/1970 00:00"
    df.loc[df["WHOIS_UPDATED_DATE"] == "b", "WHOIS_UPDATED_DATE"] = "01/01/1970 00:00"
    df.loc[df["WHOIS_UPDATED_DATE"] == "0", "WHOIS_UPDATED_DATE"] = "01/01/1970 00:00"
    df['WHOIS_UPDATED_DATE'] = df['WHOIS_UPDATED_DATE'].apply(lambda x: convert_string_to_datetime(x))

    # 2.3 Clear NaN values from numerical columns and replace with 0.0 TODO
    # Numerical columns: URL_LENGTH, NUMBER_SPECIAL_CHARACTERS, CONTENT_LENGTH, TCP_CONVERSATION_EXCHANGE,
    # DIST_REMOTE_TCP_PORT, REMOTE_IPS, APP_BYTES, SOURCE_APP_PACKETS, REMOTE_APP_PACKETS, SOURCE_APP_BYTES,
    # REMOTE_APP_BYTES, APP_PACKETS, DNS_QUERY_TIMES, Type
    df = df.fillna(0)
    # print(df.to_string())
    return df

# 3. Categorical features in two formats
# Categorical features: CHARSET, SERVER, WHOIS_COUNTRY, WHOIS_STATEPRO,
# 3.1. Dates to timespamps: WHOIS_REGDATE,WHOIS_UPDATED_DATE => these are not categorical features: why?
def convert_datetime_to_timestamps(df):
    df['WHOIS_REG_TIMESPANP'] = df['WHOIS_REGDATE'].apply(lambda x: convert_datetime_to_timestamp(x))
    df['WHOIS_UPDATED_TIMESPANP'] = df['WHOIS_UPDATED_DATE'].apply(lambda x: convert_datetime_to_timestamp(x))
    # print(df.to_string())
    return df

# 3.2 categorical 1: Supervised ratio algorithm
# Desc: ratio between the total number of rows that category present in the positive class (1 = malicious), and total number of data points
def compute_supervised_ratio_algorithm_for_column(column_name, new_column_name, column_label, df):
    list_categories = df[column_name].unique().tolist()
    df[new_column_name] = df[column_name]
    for category in list_categories:
        pi = len(df[(df[column_name]==category) &
                    (df[column_label]==1)])
        ni_and_pi = len(df[(df[column_name]==category)])
        if ni_and_pi == 0 or pi == 0:
            # print(category, pi, ni_and_pi)
            df[new_column_name] = df[new_column_name].replace(category, 0)
        else:
            df[new_column_name] = df[new_column_name].replace(category, pi/ni_and_pi)

    return df

def compute_supervised_ratio_algorithm_for_columns(df):
    df = compute_supervised_ratio_algorithm_for_column("CHARSET", "CHARSET_1", "Type", df)
    df = compute_supervised_ratio_algorithm_for_column("SERVER", "SERVER_1", "Type", df)
    df = compute_supervised_ratio_algorithm_for_column("WHOIS_COUNTRY", "WHOIS_COUNTRY_1", "Type", df)
    df = compute_supervised_ratio_algorithm_for_column("WHOIS_STATEPRO", "WHOIS_STATEPRO_1", "Type", df)
    # print(df.to_string())
    return df

# 3.3 categorical 2: weight of evidence algorithm
# ln ((Pi/TP)/(Ni/ TN))
# Pi - number of records with positive class value for the categorical attribute value in dataset
# Ni - number of records with negative class value for the categorical attribute value
# TP - total number of records with positive class value
# TN - total number of records with negative class value

def compute_weight_of_evidence_algorithm_for_column(column_name, new_column_name, column_label, df):
    list_categories = df[column_name].unique().tolist()
    df[new_column_name] = df[column_name]
    for category in list_categories:
        pi = len(df[(df[column_name]==category) &
                    (df[column_label]==1)])
        ni = len(df[(df[column_name]==category) &
                    (df[column_label]==0)])
        TN = len(df[(df[column_label]==0)])
        TP = len(df[(df[column_label]==1)])
        ratio_Pi_TP = 0
        if TP != 0:
            ratio_Pi_TP = pi/TP
        ratio_Ni_TN = 0
        if TN != 0:
            ratio_Ni_TN = ni/TN
        if ratio_Ni_TN == 0 or ratio_Pi_TP == 0:
            df[new_column_name] = df[new_column_name].replace(category, 0)
        else:
            ratio = ratio_Pi_TP/ratio_Ni_TN
            df[new_column_name] = df[new_column_name].replace(category, log(ratio, math.e))
    return df

def compute_weight_of_evidence_algorithm_for_columns(df):
    df = compute_weight_of_evidence_algorithm_for_column("CHARSET", "CHARSET_2", "Type", df)
    df = compute_weight_of_evidence_algorithm_for_column("SERVER", "SERVER_2", "Type", df)
    df = compute_weight_of_evidence_algorithm_for_column("WHOIS_COUNTRY", "WHOIS_COUNTRY_2", "Type", df)
    df = compute_weight_of_evidence_algorithm_for_column("WHOIS_STATEPRO", "WHOIS_STATEPRO_2", "Type", df)
    return df

# DROP categorical columns and URL column (which identifies each URL through an ID) and datetime columns
def remove_non_numerical_columns(df):
    df.drop(columns=['URL', 'CHARSET', 'SERVER', 'WHOIS_COUNTRY', 'WHOIS_STATEPRO',
                 'WHOIS_UPDATED_DATE', 'WHOIS_REGDATE'], inplace=True, axis = 1)
    return df
# print(df.to_string())

# 4.1 Normalize all data with MinMax Scaler and StandardScaler

def normalize_data(df, normalize = True, scaler = 'min-max'):
    """

    :param df: dataframe with dataset
    :param normalize: True or False
    :param scaler: min-max or standard
    :return: y: dataframe with labels (ground-truth)
    :return: scaled_df: dataframe with normalized dataset (if normalize_data is True, otherwise is the initial dataset)
    """
    # # Split df into X and y
    y = df['Type'].copy()
    X = df.drop('Type', axis=1).copy()

    # Normalize data along the columns
    if (normalize == True):
        if scaler == 'min-max':
            # Normalize with MinMaxScaler
            scaler = MinMaxScaler()
        elif scaler == 'standard':
            # Normalize with StandardScaler
            scaler = StandardScaler()
        names = X.columns
        d = scaler.fit_transform(X)
        scaled_df = pd.DataFrame(d, columns=names)
        # print(scaled_df.head())
    else:
        scaled_df = X.copy(deep = True)
    return y, scaled_df

# 4.2 Export normalized dataframe to CSV file
def export_dataframe_to_file(y, X, file):
    result_df = pd.concat([X, y], axis=1, join='inner')
    with open(file,'w+'):
        result_df.to_csv(file, index = False, encoding = 'utf-8')

# 5. Randomly split data into training and testing
def split_data_in_testing_training(X, y, stratify = False, train_size = 0.8):
    # Train-test split
    if (stratify == True):
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, train_size=train_size)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)

    return X_train, X_test, y_train, y_test


def export_normalized_data(df):
    y, X = normalize_data(df, normalize=True, scaler='min-max')
    export_dataframe_to_file(y, X, 'data/dataset_norm_min_max.csv')
    y, X = normalize_data(df, normalize=True, scaler='standard')
    export_dataframe_to_file(y, X, 'data/dataset_norm_standard.csv')
    y, X = normalize_data(df, normalize=False, scaler='standard')
    export_dataframe_to_file(y, X, 'data/dataset_not_norm.csv')
    # X_train, X_test, y_train, y_test = split_data_in_testing_training(y, X, stratify=True, train_size=0.8)

def load_normalized_dataset(file, normalize = False, scaler = 'min-max'):
    if (file is not None):
        df = pd.read_csv(file)
    if (normalize== False):
        df = pd.read_csv('data/dataset_not_norm.csv')
    elif scaler == 'min-max':
        df = pd.read_csv('data/dataset_norm_min_max.csv')
    else:
        df = pd.read_csv('data/dataset_norm_standard.csv')

    # # Split df into X and y
    y = df['Type'].copy()
    X = df.drop('Type', axis=1).copy()
    return y, X

# y, X = load_normalized_dataset('data/dataset_norm_standard.csv')
# print(y.to_string())
# print(X.to_string())

def main_data_preprocessing():
    df = load_dataset()
    df = clear_dataset(df)
    df = convert_datetime_to_timestamps(df)
    df = compute_supervised_ratio_algorithm_for_columns(df)
    df = compute_weight_of_evidence_algorithm_for_columns(df)
    df = remove_non_numerical_columns(df)
    export_normalized_data(df)

main_data_preprocessing()
