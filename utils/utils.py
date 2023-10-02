import os
from datetime import datetime
from time import sleep

import pandas as pd
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, roc_auc_score, classification_report, \
    matthews_corrcoef
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import time


def convert_metrics_to_csv(separator=',', *args):
    string_args = ''
    for arg in args:
        if isinstance(arg, list) or type(arg) == str or isinstance(arg, BaseEstimator):
            arg = '"' + str(arg) + '"'
        if (string_args == ''):
            string_args = str(arg)
        else:
            string_args = string_args + separator + str(arg)
    return string_args


def split_data(X, y, normalize_data=False, stratify=False, train_size=0.8, scaler='min-max'):
    # # Split df into X and y
    # y = df['label'].copy()
    # X = df.drop('label', axis=1).copy()

    # Normalize data along the columns
    # d = preprocessing.normalize(X, axis = 0)
    # scaled_df = pd.DataFrame(d, columns=names)
    # scaled_df.head()

    if (normalize_data == True):
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
        scaled_df = X.copy(deep=True)

    # Train-test split
    if (stratify == True):
        X_train, X_test, y_train, y_test = train_test_split(scaled_df, y, stratify=y, train_size=train_size)
    else:
        X_train, X_test, y_train, y_test = train_test_split(scaled_df, y, train_size=train_size)

    return X_train, X_test, y_train, y_test


def data_normalization(X, y, stratify=False, train_size=0.8):
    # Split df into X and y
    # y = df['label'].copy()
    # X = df.drop('label', axis=1).copy()

    # Normalize data along the columns
    # d = preprocessing.normalize(X, axis = 0)
    # scaled_df = pd.DataFrame(d, columns=names)
    # scaled_df.head()

    # Normalize with MinMaxScaler
    scaler = preprocessing.MinMaxScaler()
    names = X.columns
    d = scaler.fit_transform(X)
    scaled_df = pd.DataFrame(d, columns=names)

    # Train-test split
    if (stratify):
        X_train, X_test, y_train, y_test = train_test_split(scaled_df, y, stratify=y, train_size=train_size)
    else:
        X_train, X_test, y_train, y_test = train_test_split(scaled_df, y, train_size=train_size)

    return X_train, X_test, y_train, y_test

def listener_write_to_file_ensemble_withoutnewline(q, filename):
    '''listens for messages on the q, writes to file. '''
    with open(filename, 'a') as f:
        while 1:
            m = q.get()
            if m == 'kill':
                break
            f.write(m)
            f.flush()


def listener_write_to_file(q, filename):
    '''listens for messages on the q, writes to file. '''
    with open(filename, 'a') as f:
        while 1:
            m = q.get()
            if m == 'kill':
                break
            f.write(m + '\n')
            f.flush()

# Function to make predictions
def prediction(X_test, clf_object):
    # Predicton on test
    y_pred = clf_object.predict(X_test)
    y_pred_probabilities = clf_object.predict_proba(X_test)
    # print("Predicted values:")
    # print(y_pred)
    # print("Predicted probabilities: ")
    # print(y_pred_probabilities)
    return y_pred, y_pred_probabilities

# Function to calculate accuracy
def cal_metrics_general(y_test, y_pred, y_pred_probabilities):
    # print("\n-----------------------------------------------")
    # print("METRICS FOR "+ label)
    # print("-----------------------------------------------\n")

    # print("Confusion Matrix: ",
    #     confusion_matrix(y_test, y_pred))

    precision = precision_score(y_test, y_pred, pos_label=1) * 100
    # print ("Precision : ", precision)

    recall = recall_score(y_test, y_pred, pos_label=1) * 100
    # print ("Recall : ", recall)

    f1 = f1_score(y_test, y_pred, pos_label=1, labels=np.unique(y_pred)) * 100
    # print ("F1 score : ",f1)

    roc_auc = 0.0
    if (len(y_pred_probabilities) > 0):
        try:
            param_y_pred_probabilities = list(map(lambda x: x[1], y_pred_probabilities))
            roc_auc = roc_auc_score(y_test, param_y_pred_probabilities, average='weighted', labels=np.unique(y_pred))
            # print ("ROC_AUC score: ", roc_auc)
        except Exception as err:
            print(err)

    # fpr - false positive rate, tpr - true positive rate, thresholds values based on which the class 0 or 1 is chosen
    # fpr, tpr, thresholds = roc_curve(y_test, y_pred_probabilities[:,1])
    # plt.plot(fpr, tpr, marker = '.', label = "AUROC = %0.3f" %roc_auc)
    # plt.title("ROC curve")
    # plt.xlabel('False Positives Rate')
    # plt.ylabel('True POsitives Rate')
    # plt.legend()
    # plt.show()

    # print("Report : ",
    # classification_report(y_test, y_pred))

    return precision, recall, f1, roc_auc

# Function to calculate accuracy
def cal_metrics(y_test, y_pred, y_pred_probabilities, label, classifier):
    # print("\n-----------------------------------------------")
    # print("METRICS FOR "+ classifier)
    # print("-----------------------------------------------\n")

    # print("Confusion Matrix: ",
    #     confusion_matrix(y_test, y_pred))

    precision = precision_score(y_test, y_pred, pos_label=1) * 100
    # print ("Precision : ", precision)

    recall = recall_score(y_test, y_pred) * 100
    # print ("Recall : ", recall)

    f1 = f1_score(y_test, y_pred, pos_label=1, labels=np.unique(y_pred)) * 100
    # print ("F1 score : ",f1)

    roc_auc = 0.0
    if (len(y_pred_probabilities) > 0):
        try:
            roc_auc = roc_auc_score(y_test, y_pred_probabilities[:, 1], average='weighted', labels=np.unique(y_pred))
            # print ("ROC_AUC score: ", roc_auc)
        except Exception as err:
            print(err)

    # fpr - false positive rate, tpr - true positive rate, thresholds values based on which the class 0 or 1 is chosen
    # fpr, tpr, thresholds = roc_curve(y_test, y_pred_probabilities[:,1])
    # plt.plot(fpr, tpr, marker = '.', label = "AUROC = %0.3f" %roc_auc)
    # plt.title("ROC curve")
    # plt.xlabel('False Positives Rate')
    # plt.ylabel('True POsitives Rate')
    # plt.legend()
    # plt.show()

    # print("Report : ",
    # classification_report(y_test, y_pred))

    return precision, recall, f1, roc_auc


def unique(list1):
    x = np.array(list1)
    return np.unique(x)


def print_dict(dictionary):
    for k, v in dictionary.items():
        print(k, '->', v)

def current_ms() -> int:
    """
    Reports the current time in milliseconds
    :return: long int
    """
    return round(time.time() * 1000)


def preprocess_y_df(y_train_df, y_test_df):
    if isinstance(y_train_df, pd.DataFrame):
        y_train_df = y_train_df.astype('int')
        y_train = y_train_df.to_numpy().ravel()
    if isinstance(y_test_df, pd.DataFrame):
        y_test_df = y_test_df.astype('int')
        y_test = y_test_df.to_numpy().ravel()
    return y_train, y_test

def compute_statistics(start_ms, y_pred, y):
    running_time = (current_ms() - start_ms)
    report = classification_report(y, y_pred, target_names=['0', '1'], digits=4, output_dict=True)
    mcc = matthews_corrcoef(y_pred, y)
    return running_time, report, mcc


def init_results_df():
    df_results = pd.DataFrame(columns=['algorithm', 'features', 'training_time', 'testing_time', 'best_estimator',

                                       'train_accuracy', 'train_precision_0', 'train_precision_1', 'train_recall_0',
                                       'train_recall_1', 'train_f1_0', 'train_f1_1', 'train_mcc',
                                       'train_support_0', 'train_support_1', 'train_support',
                                       'train_macro_avg_precision', 'train_macro_avg_recall', 'train_macro_avg_f1',
                                       'train_weighted_avg_precision', 'train_weighted_avg_recall',
                                       'train_weighted_avg_f1',

                                       'test_accuracy', 'test_precision_0', 'test_precision_1', 'test_recall_0',
                                       'test_recall_1', 'test_f1_0', 'test_f1_1', 'test_mcc',
                                       'test_support_0', 'test_support_1', 'test_support',
                                       'test_macro_avg_precision', 'test_macro_avg_recall', 'test_macro_avg_f1',
                                       'test_weighted_avg_precision', 'test_weighted_avg_recall',
                                       'test_weighted_avg_f1',
                                       ])
    return df_results

def appendMetricsTOCSV(filename, metrics, init_function, header=False, ):
    df = pd.DataFrame(metrics)
    # append data frame to CSV file
    try:
        df.to_csv(filename, mode='a', encoding='utf-8', index=False, header=header)
    except:
        sleep(60)
        df.to_csv(filename, mode='a', encoding='utf-8', index=False, header=header)

    return init_function()


def write_error_to_log_file(e, log_file='logs.txt'):
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    my_filename = os.path.join(path_to_script, log_file)

    #   Open a file with access mode 'a'
    with open(my_filename, "a+") as file_object:
        file_object.write(f'{str(e)}\n\n')


def convert_string_to_datetime(string):
    data = None
    try:
        data = datetime.strptime(string, '%d/%m/%Y %H:%M')
    except:
        data = datetime.strptime(string, '%Y-%m-%dT%H:%M:%S.%f%z')
    return data


def convert_datetime_to_timestamp(dt):
    try:
        timestamp = int(round(dt.timestamp()))
        return timestamp
    except Exception:
        return 0
