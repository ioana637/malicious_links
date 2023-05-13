import os
from datetime import datetime
from time import sleep

import pandas as pd
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


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

    recall = recall_score(y_test, y_pred) * 100
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
