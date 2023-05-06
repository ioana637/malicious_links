import os

import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from algorithms.ada import create_label_for_ADA_from_row, create_ADA_classifier
from algorithms.dt import create_label_for_DT, create_label_for_DT_for_row, create_DT_classifier
from algorithms.enums import Algorithms, SVM_Kernels
from algorithms.knn import create_label_for_KNN, create_label_for_KNN_for_row, create_KNN_classifier
from algorithms.lr import create_label_LR_for_row, create_LR_classifier
from algorithms.nb import create_label_BNB_for_row, create_label_for_GNB_for_row, create_BNB_classifier, \
    create_GNB_classifier
from algorithms.qlda import create_label_LDA_for_row, create_LDA_classifier
from algorithms.rfc import create_label_for_rfc, create_label_for_rfc_for_row, create_RF_classifier
from algorithms.svm import create_label_SVM_for_row, create_SVM_rbf_classifier, create_SVM_poly_classifier, \
    create_SVM_sigmoid_classifier, create_SVM_linear_classifier
from algorithms.xgb import create_label_for_XGB_for_row
from data_post import compute_average_metric
from data_pre import split_data_in_testing_training, load_normalized_dataset
from utils import prediction, appendMetricsTOCSV, cal_metrics_general


def run_algorithm_ensemble_configuration(metrics, label, X, y, list_classifiers, no_classifiers=3,
                                         train_size=0.8, stratify=False):
    X_train, X_test, y_train, y_test = split_data_in_testing_training(X, y, stratify, train_size)

    y_pred_all = []
    y_pred_probabilities_all = []
    for classifier in list_classifiers:
        # Performing training
        classifier.fit(X_train, y_train)
        # Make predictions
        y_pred, y_pred_probabilities = prediction(X_test, classifier)
        y_pred_all.append(y_pred)
        y_pred_probabilities_all.append(y_pred_probabilities.tolist())

    y_pred = list(map(lambda x: round(x / no_classifiers), [sum(x) for x in zip(*y_pred_all)]))
    y_pred_probabilities = []
    length_y_pred_prob = len(y_pred_probabilities_all[0])
    for i in range(length_y_pred_prob):
        y_pred_probabilities_1 = y_pred_probabilities_all[0]
        y_pred_probabilities_2 = y_pred_probabilities_all[1]
        y_pred_probabilities_3 = y_pred_probabilities_all[2]
        v1 = round((y_pred_probabilities_1[i][0] + y_pred_probabilities_2[i][0] + y_pred_probabilities_3[i][
            0]) / no_classifiers, 2)
        v2 = round((y_pred_probabilities_1[i][1] + y_pred_probabilities_2[i][1] + y_pred_probabilities_3[i][
            1]) / no_classifiers, 2)
        y_pred_probabilities.append([v1, v2])

    # sum_prob = [np.sum(x) for x in zip(*y_pred_probabilities_all)]
    # y_pred_probabilities = list(
    #     map(lambda x: [round(x[0] / no_classifiers, 2), round(x[1] / no_classifiers, 2)], sum_prob))

    # Compute metrics
    precision, recall, f1, roc_auc = cal_metrics_general(y_test, y_pred, y_pred_probabilities)

    metrics['label'].append(label)
    metrics['precision'].append(precision)
    metrics['recall'].append(recall)
    metrics['f1_score'].append(f1)
    metrics['roc_auc'].append(roc_auc)


def init_metrics_for_ensembles():
    return {'label': [], 'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': []}


def load_df_configs_for_algs(ens: [Algorithms]):
    df_configs = {}
    for en in ens:
        if en == Algorithms.DT:
            df_config = pd.read_csv('new_results_1/dt/best.csv')
        elif en == Algorithms.KNN:
            df_config = pd.read_csv('new_results_1/knn/best.csv')
        elif en == Algorithms.RF:
            df_config = pd.read_csv('new_results_1/rfc/best.csv')
        elif en == Algorithms.ADA:
            df_config = pd.read_csv('new_results_1/ada/best.csv')
        elif en == Algorithms.LR:
            df_config = pd.read_csv('new_results_1/lr/best.csv')
        elif en == Algorithms.BNB:
            df_config = pd.read_csv('new_results_1/nb/best_bnb.csv')
        elif en == Algorithms.GNB:
            df_config = pd.read_csv('new_results_1/rfc/best_gnb.csv')
        elif en == Algorithms.LDA:
            df_config = pd.read_csv('new_results_1/lda/best.csv')
        elif en == Algorithms.SVM_rbf:
            df_config = pd.read_csv('new_results_1/svc/best-rbf.csv')
        elif en == Algorithms.SVM_linear:
            df_config = pd.read_csv('new_results_1/svc/best-lin.csv')
        elif en == Algorithms.SVM_sigmoid:
            df_config = pd.read_csv('new_results_1/svc/best-sigmoid.csv')
        elif en == Algorithms.SVM_poly:
            df_config = pd.read_csv('new_results_1/svc/best-poly.csv')
        else:
            df_config = pd.read_csv('new_results_1/xgb/best.csv')
        df_configs[en] = df_config
    return df_configs



def create_classifiers(algs, rows):
    classifiers = []
    for i in range(0, len(algs)):
        alg = algs[i]
        row = rows[i]
        if (alg == Algorithms.ADA):
            classifier = create_ADA_classifier(row)
        elif (alg == Algorithms.DT):
            classifier = create_DT_classifier(row)
        elif (alg == Algorithms.KNN):
            classifier = create_KNN_classifier(row)
        elif (alg == Algorithms.RF):
            classifier = create_RF_classifier(row)
        elif (alg == Algorithms.LR):
            classifier = create_LR_classifier(row)
        elif (alg == Algorithms.BNB):
            classifier = create_BNB_classifier(row)
        elif (alg == Algorithms.GNB):
            classifier = create_GNB_classifier(row)
        elif (alg == Algorithms.LDA):
            classifier = create_LDA_classifier(row)
        elif (alg == Algorithms.XGB):
            classifier = create_XGB_classifier(row)
        elif (alg == Algorithms.SVM_rbf):
            classifier = create_SVM_rbf_classifier(row)
        elif (alg == Algorithms.SVM_poly):
            classifier = create_SVM_poly_classifier(row)
        elif (alg == Algorithms.SVM_sigmoid):
            classifier = create_SVM_sigmoid_classifier(row)
        elif (alg == Algorithms.SVM_linear):
            classifier = create_SVM_linear_classifier(row)
        else:
            classifier = None
        classifiers.append(classifier)
    return classifiers


def run_algorithm_ensemble(filename='', path='', stratify=True, train_size=0.8,
                           normalize_data=True, scaler='min-max', no_repeats=100, ens: [Algorithms] = None,
                           ):
    if ens == None or len(ens) < 3:
        return
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_ensembles()

    my_filename = os.path.join(path, 'results\\ens', filename)

    # df_configs e dictionar
    df_configs = load_df_configs_for_algs(ens)
    for alg, df_config in df_configs.items():
        print(alg, '->', df_config.head())

    algs = list(df_configs.keys())
    df_configs_values = df_configs.values()
    df_config_1 = df_configs[algs[0]]
    df_config_2 = df_configs[algs[1]]
    df_config_3 = df_configs[algs[2]]

    for index_1, row_1 in df_config_1.iterrows():
        for index_2, row_2 in df_config_2.iterrows():
            for index_3, row_3 in df_config_3.iterrows():
                for i in range(0, no_repeats):
                    label = create_label_for_ensemble(algs, [row_1, row_2, row_3])
                    classifiers = create_classifiers(algs, [row_1, row_2, row_3])
                    run_algorithm_ensemble_configuration(metrics, label, X, y, classifiers, len(classifiers),
                                                         train_size=train_size, stratify=stratify)

    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.groupby(['label'], as_index=False).agg(
        {'precision': 'mean', 'recall': 'mean', 'f1_score': 'mean', 'roc_auc': 'mean'})
    metrics_df = compute_average_metric(metrics_df)
    metrics_df.sort_values(by=['average_metric'], ascending=False, inplace=True)
    metrics = appendMetricsTOCSV(my_filename, metrics_df, init_metrics_for_ensembles, header=True)


# def run_algorithm_ensemble_dt_knn_rf(filename='', path='', stratify=True, train_size=0.8,
#                                      normalize_data=True, scaler='min-max', no_repeats=100):
#     y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
#     metrics = init_metrics_for_ensembles()
#
#     my_filename = os.path.join(path, 'results\\ens', filename)
#
#     df_configs_dt = pd.read_csv('new_results_1/dt/best.csv')
#     df_configs_knn = pd.read_csv('new_results_1/knn/best.csv')
#     df_configs_rf = pd.read_csv('new_results_1/rfc/best.csv')
#
#     for index_rf, row_rf in df_configs_rf.iterrows():
#         for index_dt, row_dt in df_configs_dt.iterrows():
#             for index_knn, row_knn in df_configs_knn.iterrows():
#                 for i in range(0, no_repeats):
#                     label = create_label_for_ensemble_DT_KNN_RF(row_dt, row_knn, row_rf)
#                     classifier1 = create_DT_classifier(row_dt)
#                     classifier2 = create_KNN_classifier(row_knn)
#                     classifier3 = create_RF_classifier(row_rf)
#                     run_algorithm_ensemble_configuration(metrics, label, X, y,
#                                                          [classifier1, classifier2, classifier3], 3,
#                                                          train_size=train_size, stratify=stratify)
#
#     metrics_df = pd.DataFrame(metrics)
#     metrics_df = metrics_df.groupby(['label'], as_index=False).agg(
#         {'precision': 'mean', 'recall': 'mean', 'f1_score': 'mean', 'roc_auc': 'mean'})
#     metrics_df = compute_average_metric(metrics_df)
#     metrics_df.sort_values(by=['average_metric'], ascending=False, inplace=True)
#     metrics = appendMetricsTOCSV(my_filename, metrics_df, init_metrics_for_ensembles, header=True)


def create_label_for_ensemble(algs: [Algorithms], rows: []):
    label = "ENSEMBLE "
    for alg in algs:
        label += str(alg.value).upper() + " "
    for i in range(0, len(algs)):
        alg = algs[i]
        row = rows[i].value
        if (alg == Algorithms.ADA):
            label += create_label_for_ADA_from_row(row)
        elif (alg == Algorithms.DT):
            label += create_label_for_DT_for_row(row)
        elif (alg == Algorithms.KNN):
            label += create_label_for_KNN_for_row(row)
        elif (alg == Algorithms.RF):
            label += create_label_for_rfc_for_row(row)
        elif (alg == Algorithms.LR):
            label += create_label_LR_for_row(row)
        elif (alg == Algorithms.BNB):
            label += create_label_BNB_for_row(row)
        elif (alg == Algorithms.GNB):
            label += create_label_for_GNB_for_row(row)
        elif (alg == Algorithms.LDA):
            label += create_label_LDA_for_row(row)
        elif (alg == Algorithms.XGB):
            label += create_label_for_XGB_for_row(row)
        elif (alg == Algorithms.SVM_rbf):
            label += create_label_SVM_for_row(SVM_Kernels.rbf, row)
        elif (alg == Algorithms.SVM_poly):
            label += create_label_SVM_for_row(SVM_Kernels.poly, row)
        elif (alg == Algorithms.SVM_sigmoid):
            label += create_label_SVM_for_row(SVM_Kernels.sigmoid, row)
        elif (alg == Algorithms.SVM_linear):
            label += create_label_SVM_for_row(SVM_Kernels.linear, row)
