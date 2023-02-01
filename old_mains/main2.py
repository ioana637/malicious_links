import os
from datetime import datetime

import pandas as pd

from ada import run_algorithm_ada_boost
from gpc import run_algorithm_gpc
from lr import run_algorithm_lr
from nb import run_algorithm_gnb, run_algorithm_bnb
from qlda import run_algorithm_lda, run_algorithm_qda
from svm import run_algorithm_SVC_linear_kernel, run_algorithm_SVC_RBF_kernel, run_algorithm_SVC_sigmoid_kernel
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

# ----------------------LOGISTIC REGRESSION --------------------------------

# for train_size in [0.8]:
    # run_algorithm_rf(df_preprocessed, 'metrics_RFC_train_size_'+str(int(train_size * 100))+'_without_stratify.csv', stratify = False, train_size = train_size)
    # run_algorithm_rf(df_preprocessed, 'metrics_RFC_train_size_'+str(int(train_size * 100))+'_with_stratify.csv', stratify = True, train_size = train_size)
    # run_algorithm_lr(df_preprocessed, 'metrics_DN_min_max_LR_train_size_' + str(
    #     int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv',
    #                  stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')
    # run_algorithm_lr(df_preprocessed, 'metrics_DN_standard_LR_train_size_' + str(
    #     int(train_size * 100)) + '_with_stratify_' + dt_string  + '.csv',
    #                  stratify=True, train_size=train_size, normalize_data=True, scaler='standard')
# run_algorithm_rf(df_preprocessed, 'metrics_DN_standard_RFC_train_size_'+str(int(train_size * 100))+'_without_stratify.csv',
#                  stratify = False, train_size = train_size, normalize_data = True, scaler = 'standard')
# run_algorithm_rf(df_preprocessed, 'metrics_DN_min_max_RFC_train_size_'+str(int(train_size * 100))+'_without_stratify.csv',
#                  stratify = False, train_size = train_size, normalize_data = True, scaler = 'min-max')


# -----------------------NAIVE BAYES------------------------
# for train_size in [0.8]:
#     # run_algorithm_rf(df_preprocessed, 'metrics_RFC_train_size_'+str(int(train_size * 100))+'_without_stratify.csv', stratify = False, train_size = train_size)
#     # run_algorithm_rf(df_preprocessed, 'metrics_RFC_train_size_'+str(int(train_size * 100))+'_with_stratify.csv', stratify = True, train_size = train_size)
#     run_algorithm_gnb(df_preprocessed, 'metrics_DN_min_max_GNB_train_size_' + str(
#         int(train_size * 100)) + '_with_stratify.csv',
#                      stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')
#     run_algorithm_gnb(df_preprocessed, 'metrics_DN_standard_GNB_train_size_' + str(
#         int(train_size * 100)) + '_with_stratify.csv',
#                      stratify=True, train_size=train_size, normalize_data=True, scaler='standard')
# run_algorithm_rf(df_preprocessed, 'metrics_DN_standard_RFC_train_size_'+str(int(train_size * 100))+'_without_stratify.csv',
#                  stratify = False, train_size = train_size, normalize_data = True, scaler = 'standard')
# run_algorithm_rf(df_preprocessed, 'metrics_DN_min_max_RFC_train_size_'+str(int(train_size * 100))+'_without_stratify.csv',
#                  stratify = False, train_size = train_size, normalize_data = True, scaler = 'min-max')

# for train_size in [0.8]:
#     # run_algorithm_rf(df_preprocessed, 'metrics_RFC_train_size_'+str(int(train_size * 100))+'_without_stratify.csv', stratify = False, train_size = train_size)
#     # run_algorithm_rf(df_preprocessed, 'metrics_RFC_train_size_'+str(int(train_size * 100))+'_with_stratify.csv', stratify = True, train_size = train_size)
#     run_algorithm_bnb(df_preprocessed, 'metrics_DN_min_max_BNB_train_size_' + str(
#         int(train_size * 100)) + '_with_stratify.csv',
#                       stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')
#     run_algorithm_bnb(df_preprocessed, 'metrics_DN_standard_BNB_train_size_' + str(
#         int(train_size * 100)) + '_with_stratify.csv',
#                       stratify=True, train_size=train_size, normalize_data=True, scaler='standard')
# run_algorithm_rf(df_preprocessed, 'metrics_DN_standard_RFC_train_size_'+str(int(train_size * 100))+'_without_stratify.csv',
#                  stratify = False, train_size = train_size, normalize_data = True, scaler = 'standard')
# run_algorithm_rf(df_preprocessed, 'metrics_DN_min_max_RFC_train_size_'+str(int(train_size * 100))+'_without_stratify.csv',
#                  stratify = False, train_size = train_size, normalize_data = True, scaler = 'min-max')

# TODO Categorical NB *** are ceva erori
# TODO Complement Naive Bayes
# TODO MultinomialNB


# ----------------------SVMs--------------------------------------
# SVM with rbf kernel
# for train_size in [0.8]:
    # run_algorithm_rf(df_preprocessed, 'metrics_RFC_train_size_'+str(int(train_size * 100))+'_without_stratify.csv', stratify = False, train_size = train_size)
    # run_algorithm_rf(df_preprocessed, 'metrics_RFC_train_size_'+str(int(train_size * 100))+'_with_stratify.csv', stratify = True, train_size = train_size)
    # run_algorithm_SVC_RBF_kernel(df_preprocessed, 'metrics_DN_min_max_SVC_RBF_kernel_train_size_' + str(
    #     int(train_size * 100)) + '_with_stratify.csv',
    #                                 stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')
    # run_algorithm_SVC_RBF_kernel(df_preprocessed, 'metrics_DN_standard_SVC_RBF_kernel_train_size_' + str(
    #     int(train_size * 100)) + '_with_stratify.csv',
    #                                 stratify=True, train_size=train_size, normalize_data=True, scaler='standard')
# run_algorithm_rf(df_preprocessed, 'metrics_DN_standard_RFC_train_size_'+str(int(train_size * 100))+'_without_stratify.csv',
#                  stratify = False, train_size = train_size, normalize_data = True, scaler = 'standard')
# run_algorithm_rf(df_preprocessed, 'metrics_DN_min_max_RFC_train_size_'+str(int(train_size * 100))+'_without_stratify.csv',
#                  stratify = False, train_size = train_size, normalize_data = True, scaler = 'min-max')



# SVM with polynomial kernel
# TODO computation very expensive!!!

# SVM with linear
# for train_size in [0.8]:
    # run_algorithm_rf(df_preprocessed, 'metrics_RFC_train_size_'+str(int(train_size * 100))+'_without_stratify.csv', stratify = False, train_size = train_size)
    # run_algorithm_rf(df_preprocessed, 'metrics_RFC_train_size_'+str(int(train_size * 100))+'_with_stratify.csv', stratify = True, train_size = train_size)
    # run_algorithm_SVC_linear_kernel(df_preprocessed, 'metrics_DN_min_max_SVC_linear_kernel_train_size_' + str(
    #     int(train_size * 100)) + '_with_stratify.csv',
    #                   stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')
    # run_algorithm_SVC_linear_kernel(df_preprocessed, 'metrics_DN_standard_SVC_linear_kernel_train_size_' + str(
    #     int(train_size * 100)) + '_with_stratify.csv',
    #                   stratify=True, train_size=train_size, normalize_data=True, scaler='standard')
# run_algorithm_rf(df_preprocessed, 'metrics_DN_standard_RFC_train_size_'+str(int(train_size * 100))+'_without_stratify.csv',
#                  stratify = False, train_size = train_size, normalize_data = True, scaler = 'standard')
# run_algorithm_rf(df_preprocessed, 'metrics_DN_min_max_RFC_train_size_'+str(int(train_size * 100))+'_without_stratify.csv',
#                  stratify = False, train_size = train_size, normalize_data = True, scaler = 'min-max')


# SVM with sigmoid
for train_size in [0.8]:
    # run_algorithm_rf(df_preprocessed, 'metrics_RFC_train_size_'+str(int(train_size * 100))+'_without_stratify.csv', stratify = False, train_size = train_size)
    # run_algorithm_rf(df_preprocessed, 'metrics_RFC_train_size_'+str(int(train_size * 100))+'_with_stratify.csv', stratify = True, train_size = train_size)
    run_algorithm_SVC_sigmoid_kernel(df_preprocessed, 'metrics_DN_min_max_SVC_sigmoid_kernel_train_size_' + str(
        int(train_size * 100)) + '_with_stratify.csv',
                                    stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')
    # run_algorithm_SVC_sigmoid_kernel(df_preprocessed, 'metrics_DN_standard_SVC_sigmoid_kernel_train_size_' + str(
    #     int(train_size * 100)) + '_with_stratify.csv',
    #                                 stratify=True, train_size=train_size, normalize_data=True, scaler='standard')
# run_algorithm_rf(df_preprocessed, 'metrics_DN_standard_RFC_train_size_'+str(int(train_size * 100))+'_without_stratify.csv',
#                  stratify = False, train_size = train_size, normalize_data = True, scaler = 'standard')
# run_algorithm_rf(df_preprocessed, 'metrics_DN_min_max_RFC_train_size_'+str(int(train_size * 100))+'_without_stratify.csv',
#                  stratify = False, train_size = train_size, normalize_data = True, scaler = 'min-max')





# -------------------GAUSSIAN PROCESS CLASSIFIER-------------------
# for train_size in [0.8]:
#     # run_algorithm_rf(df_preprocessed, 'metrics_RFC_train_size_'+str(int(train_size * 100))+'_without_stratify.csv', stratify = False, train_size = train_size)
#     # run_algorithm_rf(df_preprocessed, 'metrics_RFC_train_size_'+str(int(train_size * 100))+'_with_stratify.csv', stratify = True, train_size = train_size)
#     run_algorithm_gpc(df_preprocessed, 'metrics_DN_min_max_GPC_train_size_' + str(
#         int(train_size * 100)) + '_with_stratify.csv',
#                             stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')
#     run_algorithm_gpc(df_preprocessed, 'metrics_DN_standard_GPC_train_size_' + str(
#         int(train_size * 100)) + '_with_stratify.csv',
#                             stratify=True, train_size=train_size, normalize_data=True, scaler='standard')
# run_algorithm_rf(df_preprocessed, 'metrics_DN_standard_RFC_train_size_'+str(int(train_size * 100))+'_without_stratify.csv',
#                  stratify = False, train_size = train_size, normalize_data = True, scaler = 'standard')
# run_algorithm_rf(df_preprocessed, 'metrics_DN_min_max_RFC_train_size_'+str(int(train_size * 100))+'_without_stratify.csv',
#                  stratify = False, train_size = train_size, normalize_data = True, scaler = 'min-max')




#------------------------QUADRATIC & LINEAR DISCRIMANT ANALYSIS------------------------
# for train_size in [0.8]:
#     run_algorithm_rf(df_preprocessed, 'metrics_RFC_train_size_'+str(int(train_size * 100))+'_without_stratify.csv', stratify = False, train_size = train_size)
    # run_algorithm_rf(df_preprocessed, 'metrics_RFC_train_size_'+str(int(train_size * 100))+'_with_stratify.csv', stratify = True, train_size = train_size)
    # run_algorithm_lda(df_preprocessed, 'metrics_DN_min_max_LDA_train_size_' + str(
    #     int(train_size * 100)) + '_with_stratify.csv',
    #                                  stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')
    # run_algorithm_lda(df_preprocessed, 'metrics_DN_standard_LDA_train_size_' + str(
    #     int(train_size * 100)) + '_with_stratify.csv',
    #                                  stratify=True, train_size=train_size, normalize_data=True, scaler='standard')
# run_algorithm_rf(df_preprocessed, 'metrics_DN_standard_RFC_train_size_'+str(int(train_size * 100))+'_without_stratify.csv',
#                  stratify = False, train_size = train_size, normalize_data = True, scaler = 'standard')
# run_algorithm_rf(df_preprocessed, 'metrics_DN_min_max_RFC_train_size_'+str(int(train_size * 100))+'_without_stratify.csv',
#                  stratify = False, train_size = train_size, normalize_data = True, scaler = 'min-max')

# for train_size in [0.8]:
    # run_algorithm_rf(df_preprocessed, 'metrics_RFC_train_size_'+str(int(train_size * 100))+'_without_stratify.csv', stratify = False, train_size = train_size)
    # run_algorithm_rf(df_preprocessed, 'metrics_RFC_train_size_'+str(int(train_size * 100))+'_with_stratify.csv', stratify = True, train_size = train_size)
    # run_algorithm_qda(df_preprocessed, 'metrics_DN_min_max_QDA_train_size_' + str(
    #     int(train_size * 100)) + '_with_stratify.csv',
    #                   stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')
    # run_algorithm_qda(df_preprocessed, 'metrics_DN_standard_QDA_train_size_' + str(
    #     int(train_size * 100)) + '_with_stratify.csv',
    #                   stratify=True, train_size=train_size, normalize_data=True, scaler='standard')
# run_algorithm_rf(df_preprocessed, 'metrics_DN_standard_RFC_train_size_'+str(int(train_size * 100))+'_without_stratify.csv',
#                  stratify = False, train_size = train_size, normalize_data = True, scaler = 'standard')
# run_algorithm_rf(df_preprocessed, 'metrics_DN_min_max_RFC_train_size_'+str(int(train_size * 100))+'_without_stratify.csv',
#                  stratify = False, train_size = train_size, normalize_data = True, scaler = 'min-max')

