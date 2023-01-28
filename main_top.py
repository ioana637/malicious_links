import os
import pandas as pd

from dt import run_algorithm_dt, run_top_20_DT_configs
from knn import run_algorithm_KNN, run_top_20_KNN_configs
from rfc import run_algorithm_rf, run_top_20_RFC_configs
from datetime import datetime

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


def loadDataFromDatasetFile():
    inputFile = r'data\dataset.csv'
    cwd = os.getcwd()
    df = pd.read_csv(f'{cwd}\\{inputFile}')
    print(df.to_string())
    df = df.drop(['CHARSET1', 'SERVER1', 'WHOIS_COUNTRY1', 'WHOIS_STATEPRO1', 'WHOIS_REGDATE1',
                  'WHOIS_UPDATED_DATE1', 'TCP_CONVERSATION_EXCHANGE1', 'DIST_REMOTE_TCP_PORT1',
                  'REMOTE_IPS1', 'APP_BYTES1', 'SOURCE_APP_PACKETS1', 'REMOTE_APP_PACKETS1',
                  'SOURCE_APP_BYTES1', 'REMOTE_APP_BYTES1', 'APP_PACKETS1', 'DNS_QUERY_TIMES1'])

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
# df_preprocessed = loadDataFromDatasetFile()


# ----------------DECISION TREE----------------------------------------------------
run_top_20_DT_configs(df_preprocessed, filename = 'top20_DT_train_size_80_with_stratify_DN_minmax.csv', scaler='min-max', n_rep=100)
run_top_20_DT_configs(df_preprocessed, filename = 'top20_DT_train_size_80_with_stratify_DN_standard.csv', scaler='standard', n_rep=100)

# ---------------------K-Nearest Neighbor ---------------------------------
run_top_20_KNN_configs(df_preprocessed, filename = 'top20_KNN_train_size_80_with_stratify_DN_minmax.csv', scaler='min-max', n_rep=100)
run_top_20_KNN_configs(df_preprocessed, filename = 'top20_KNN_train_size_80_with_stratify_DN_standard.csv', scaler='standard', n_rep=100)

# ------------------------RANDOM FOREST ------------------------------------------
run_top_20_RFC_configs(df_preprocessed, filename = 'top20_RF_train_size_80_with_stratify_DN_minmax.csv', scaler='min-max', n_rep=100)
run_top_20_RFC_configs(df_preprocessed, filename = 'top20_RF_train_size_80_with_stratify_DN_standard.csv', scaler='standard', n_rep=100)
