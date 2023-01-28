import os
import pandas as pd

from dt import run_algorithm_dt
from knn import run_algorithm_KNN
from rfc import run_algorithm_rf
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

# ----------------------Multi Layer Perceptron --------------------------------
# RUN algorithm without data normalization

# for train_size in [0.2, 0.3, 0.4,0.5,0.6,0.7,0.8]:
# for train_size in [0.2]:
#     run_algorithm_mlp(df_preprocessed, filename = 'metrics_MLP_train_size_'+str(int(train_size*100))+'_with_stratify.csv', stratify = True, train_size = train_size, normalize_data=False)
#     run_algorithm_mlp(df_preprocessed, filename = 'metrics_MLP_train_size_'+str(int(train_size*100))+'_without_stratify.csv', stratify = False, train_size = train_size, normalize_data=False)


# RUN algorithm WITH data normalization
# for train_size in [0.2]:
#     run_algorithm_mlp(df_preprocessed, filename = 'metrics_DN_MLP_train_size_'+str(int(train_size*100))+'_with_stratify.csv', stratify = True, train_size = train_size, normalize_data=True)
#     run_algorithm_mlp(df_preprocessed, filename = 'metrics_DN_MLP_train_size_'+str(int(train_size*100))+'_without_stratify.csv', stratify = False, train_size = train_size, normalize_data=True)

# -------------------------------Support Vector Machine with Polynomial kernel ------------------------
# RUN algorithm WITHOUT data normalization
# for train_size in [0.2]:
#     run_algorithm_SVC_poly_kernel(df_preprocessed, 'metrics_SVC_poly_kernel_train_size_'+str(int(train_size * 100))+'_without_stratify.csv', stratify = False, train_size = train_size, normalize_data=False)
#     # run_algorithm_SVC_poly_kernel(df_preprocessed, 'metrics_SVC_poly_kernel_train_size_'+str(int(train_size * 100))+'_with_stratify.csv', stratify = True, train_size = train_size, normalize_data= False)
#
# # RUN algorithm WITH data normalization
# for train_size in [0.2]:
#     # run_algorithm_SVC_poly_kernel(df_preprocessed, 'metrics_DN_SVC_poly_kernel_train_size_'+str(int(train_size * 100))+'_without_stratify.csv', stratify = False, train_size = train_size, normalize_data=True)
#     run_algorithm_SVC_poly_kernel(df_preprocessed, 'metrics_DN_SVC_poly_kernel_train_size_'+str(int(train_size * 100))+'_with_stratify.csv', stratify = True, train_size = train_size, normalize_data=True)

# ---------------------------------Random Forest ---------------------------------------
for train_size in [0.2, 0.3, 0.4, 0.5, 0.6]:
# run_algorithm_rf(df_preprocessed, 'metrics_RFC_train_size_'+str(int(train_size * 100))+'_without_stratify.csv', stratify = False, train_size = train_size)
# run_algorithm_rf(df_preprocessed, 'metrics_RFC_train_size_'+str(int(train_size * 100))+'_with_stratify.csv', stratify = True, train_size = train_size)
    run_algorithm_rf(df_preprocessed, 'metrics_DN_min_max_RFC_train_size_' + str(
        int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv',
                     stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')
    run_algorithm_rf(df_preprocessed, 'metrics_DN_standard_RFC_train_size_' + str(
        int(train_size * 100)) + '_with_stratify_' + dt_string  + '.csv',
                     stratify=True, train_size=train_size, normalize_data=True, scaler='standard')
# run_algorithm_rf(df_preprocessed, 'metrics_DN_standard_RFC_train_size_'+str(int(train_size * 100))+'_without_stratify.csv',
#                  stratify = False, train_size = train_size, normalize_data = True, scaler = 'standard')
# run_algorithm_rf(df_preprocessed, 'metrics_DN_min_max_RFC_train_size_'+str(int(train_size * 100))+'_without_stratify.csv',
#                  stratify = False, train_size = train_size, normalize_data = True, scaler = 'min-max')

# ----------------DECISION TREE----------------------------------------------------
# run_algorithm_dt(df_preprocessed, file_name = 'metrics_DT_train_size_70_without_stratify', stratify = False, train_size=0.7)
# run_algorithm_dt(df_preprocessed, file_name = 'metrics_DT_train_size_80_without_stratify', stratify = False, train_size=0.8)

# for train_size in [0.2,0.3,0.4,0.5,0.6,0.7,0.8]:
#   run_algorithm_dt(df_preprocessed, file_name = 'metrics_DT_train_size_'+str(int(train_size * 100))+'_with_stratify.csv', stratify = True, train_size=train_size)


# for train_size in [0.8]:
# run_algorithm_dt(df_preprocessed, 'metrics_DT_train_size_'+str(int(train_size * 100))+'_without_stratify.csv', stratify = False, train_size = train_size, normalize_data = False)
# run_algorithm_dt(df_preprocessed, 'metrics_DT_train_size_'+str(int(train_size * 100))+'_with_stratify.csv', stratify = True, train_size = train_size, normalize_data = False)
# run_algorithm_dt(df_preprocessed,
#                  'metrics_DN_min_max_DT_train_size_' + str(int(train_size * 100)) + '_with_stratify.csv',
#                  stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')
# run_algorithm_dt(df_preprocessed,
#                  'metrics_DN_standard_DT_train_size_' + str(int(train_size * 100)) + '_with_stratify.csv',
#                  stratify=True, train_size=train_size, normalize_data=True, scaler='standard')
# run_algorithm_dt(df_preprocessed, 'metrics_DN_min_max_DT_train_size_'+str(int(train_size * 100))+'_without_stratify.csv', stratify = False, train_size = train_size, normalize_data = True, scaler = 'min-max')
# run_algorithm_dt(df_preprocessed, 'metrics_DN_standard_DT_train_size_'+str(int(train_size * 100))+'_without_stratify.csv', stratify = False, train_size = train_size, normalize_data = True, scaler = 'standard')

# ---------------------K-Nearest Neighbor ---------------------------------

# for train_size in [0.5]:
    #     # run_algorithm_KNN(df_preprocessed, filename = 'metrics_KNN_train_size_'+str(int(train_size*100))+'_with_stratify.csv', stratify = True, train_size = train_size)
    #     # run_algorithm_KNN(df_preprocessed, filename = 'metrics_KNN_train_size_'+str(int(train_size*100))+'_without_stratify.csv', stratify = False, train_size = train_size)
    # run_algorithm_KNN(df_preprocessed,
    #                   'metrics_DN_min_max_KNN_train_size_' + str(int(train_size * 100)) + '_with_stratify.csv',
    #                   stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')
    # run_algorithm_KNN(df_preprocessed,
    #                   'metrics_DN_standard_KNN_train_size_' + str(int(train_size * 100)) + '_with_stratify.csv',
    #                   stratify=True, train_size=train_size, normalize_data=True, scaler='standard')
#     # run_algorithm_KNN(df_preprocessed, 'metrics_DN_standard_KNN_train_size_'+str(int(train_size * 100))+'_without_stratify.csv',
#     #  stratify = False, train_size = train_size, normalize_data = True, scaler = 'standard')
#     # run_algorithm_KNN(df_preprocessed, 'metrics_DN_min_max_KNN_train_size_'+str(int(train_size * 100))+'_without_stratify.csv',
#     #  stratify = False, train_size = train_size, normalize_data = True, scaler = 'min-max')
