import os
from datetime import datetime

import pandas as pd

from ada import run_algorithm_ada_boost
from xgb import run_algorithm_xgb

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M")


def loadDataOutput():
    inputFile = r'data\output_1.csv'
    cwd = os.getcwd()
    df = pd.read_csv(f'{cwd}\\{inputFile}')
    df["features"] = df["features"].str.replace('[', '', regex=True)
    df["features"] = df["features"].str.replace(']', '', regex=True)
    df[['index', 'url_length',
        'number_special_characters', 'charset',
        'server', 'content_length', 'whois_country',
        'whois_statepro', 'whois_regdate',
        'whois_updated_date', 'tcp_conversation_exchange',
        'dist_remote_tcp_port', 'remote_ips', 'app_bytes',
        'source_app_packets', 'remote_app_packets',
        'source_app_bytes', 'remote_app_bytes',
        'app_packets', 'dns_query_times']] = df["features"].str.split(',', expand=True)
    df = df.drop('features', axis=1)
    df = df.drop('index', axis=1)

    count_class_balance = df.apply(lambda x: True if x['label'] == 0 else False, axis=1)
    #   # 0 - malicious; 1 - benign
    #   # False for benign and True for malicious
    # # Count number of True in the series
    num_rows_malicious = len(count_class_balance[count_class_balance == True].index)
    print('num_rows_malicious', num_rows_malicious)
    num_rows_benign = len(count_class_balance[count_class_balance == False].index)
    print('num_rows_benign', num_rows_benign)

    df_preprocessed = df.copy(deep=True)
    # df_normalized = df.copy(deep=True)
    return df_preprocessed

df_preprocessed = loadDataOutput()





# -----------------------ADA BOOST-----------------------------

for train_size in [0.8]:
    # run_algorithm_rf(df_preprocessed, 'metrics_RFC_train_size_'+str(int(train_size * 100))+'_without_stratify.csv', stratify = False, train_size = train_size)
    # run_algorithm_rf(df_preprocessed, 'metrics_RFC_train_size_'+str(int(train_size * 100))+'_with_stratify.csv', stratify = True, train_size = train_size)
    run_algorithm_ada_boost(df_preprocessed, 'metrics_DN_min_max_ADA_BOOST_train_size_' + str(
        int(train_size * 100)) + '_with_stratify.csv',
                            stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')
    run_algorithm_ada_boost(df_preprocessed, 'metrics_DN_standard_ADA_BOOST_train_size_' + str(
        int(train_size * 100)) + '_with_stratify.csv',
                            stratify=True, train_size=train_size, normalize_data=True, scaler='standard')
# run_algorithm_rf(df_preprocessed, 'metrics_DN_standard_RFC_train_size_'+str(int(train_size * 100))+'_without_stratify.csv',
#                  stratify = False, train_size = train_size, normalize_data = True, scaler = 'standard')
# run_algorithm_rf(df_preprocessed, 'metrics_DN_min_max_RFC_train_size_'+str(int(train_size * 100))+'_without_stratify.csv',
#                  stratify = False, train_size = train_size, normalize_data = True, scaler = 'min-max')


# _____________________GRADIENT BOOST---------------------------------
for train_size in [0.8]:
    # run_algorithm_rf(df_preprocessed, 'metrics_RFC_train_size_'+str(int(train_size * 100))+'_without_stratify.csv', stratify = False, train_size = train_size)
    # run_algorithm_rf(df_preprocessed, 'metrics_RFC_train_size_'+str(int(train_size * 100))+'_with_stratify.csv', stratify = True, train_size = train_size)
    run_algorithm_xgb(df_preprocessed, 'metrics_DN_min_max_XGBOOST_train_size_' + str(
        int(train_size * 100)) + '_with_stratify.csv',
                      stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')
    run_algorithm_xgb(df_preprocessed, 'metrics_DN_standard_XGBOOST_train_size_' + str(
        int(train_size * 100)) + '_with_stratify.csv',
                      stratify=True, train_size=train_size, normalize_data=True, scaler='standard')
# run_algorithm_rf(df_preprocessed, 'metrics_DN_standard_RFC_train_size_'+str(int(train_size * 100))+'_without_stratify.csv',
#                  stratify = False, train_size = train_size, normalize_data = True, scaler = 'standard')
# run_algorithm_rf(df_preprocessed, 'metrics_DN_min_max_RFC_train_size_'+str(int(train_size * 100))+'_without_stratify.csv',
#                  stratify = False, train_size = train_size, normalize_data = True, scaler = 'min-max')


