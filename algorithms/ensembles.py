import os

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

from algorithms.ada import create_label_for_ADA_from_row, create_ADA_classifier
from algorithms.dt import create_label_for_DT_for_row, create_DT_classifier
from algorithms.enums import Algorithms, SVM_Kernels
from algorithms.gpc import create_GPC_classifier
from algorithms.knn import create_label_for_KNN_for_row, create_KNN_classifier
from algorithms.lr import create_label_LR_for_row, create_LR_classifier
from algorithms.mlp import create_MLP_classifier
from algorithms.nb import create_label_BNB_for_row, create_label_for_GNB_for_row, create_BNB_classifier, \
    create_GNB_classifier
from algorithms.qlda import create_label_LDA_for_row, create_LDA_classifier
from algorithms.rfc import create_label_for_rfc_for_row, create_RF_classifier
from algorithms.svm import create_label_SVM_for_row, create_SVM_classifier
from algorithms.xgb import create_label_for_XGB_for_row, create_XGB_classifier
from utils.data_post import compute_average_metric, compute_roc_auc_score_100
from utils.data_pre import split_data_in_testing_training, load_normalized_dataset
from utils.utils import prediction, appendMetricsTOCSV, cal_metrics_general

np.set_printoptions(linewidth=100000)


@ignore_warnings(category=ConvergenceWarning)
def run_algorithm_ensemble_configuration(metrics, label, ensemble, X, y, list_classifiers, no_classifiers=3,
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
    metrics['ensemble'].append(ensemble)
    metrics['precision'].append(precision)
    metrics['recall'].append(recall)
    metrics['f1_score'].append(f1)
    metrics['roc_auc'].append(roc_auc)


def init_metrics_for_ensembles():
    return {'label': [], 'ensemble': [], 'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': []}


def load_df_configs_for_algs(ens: [Algorithms]):
    df_configs = {}
    for en in ens:
        if en == Algorithms.DT:
            # df_config = pd.read_csv('new_results/dt/best-dt-1.csv')
            df_config = pd.read_csv('new_results\\dt\\best-dt-macs.csv')
        elif en == Algorithms.KNN:
            # df_config = pd.read_csv('new_results/knn/best-knn-1.csv')
            df_config = pd.read_csv('new_results\\knn\\best-knn-macs.csv')
        elif en == Algorithms.RF:
            # df_config = pd.read_csv('new_results/rfc/best-rfc-1.csv')
            df_config = pd.read_csv('new_results\\rfc\\best-rfc-macs.csv')
        elif en == Algorithms.ADA:
            # df_config = pd.read_csv('new_results/ada/best-ada-1.csv')
            df_config = pd.read_csv('new_results\\ada\\best-ada-1.csv')
        elif en == Algorithms.LR:
            # df_config = pd.read_csv('new_results/lr/best-lr-1.csv')
            df_config = pd.read_csv('new_results\\lr\\best-lr-1.csv')
        elif en == Algorithms.BNB:
            # df_config = pd.read_csv('new_results/nb/best-bnb-1.csv')
            df_config = pd.read_csv('new_results\\nb\\best-bnb-1.csv')
        elif en == Algorithms.GNB:
            # df_config = pd.read_csv('new_results/rfc/best-gnb-1.csv')
            df_config = pd.read_csv('new_results\\rfc\\best-gnb-1.csv')
        elif en == Algorithms.LDA:
            # df_config = pd.read_csv('new_results/qlda/best-lda-1.csv')
            df_config = pd.read_csv('new_results\\qlda\\best-lda-1.csv')
        elif en == Algorithms.SVM_rbf:
            # df_config = pd.read_csv('new_results/svc/best-rbf-1.csv')
            df_config = pd.read_csv('new_results\\svc\\best-rbf-1.csv')
        elif en == Algorithms.SVM_linear:
            # df_config = pd.read_csv('new_results/svc/best-linear-1.csv')
            df_config = pd.read_csv('new_results\\svc\\best-linear-1.csv')
        elif en == Algorithms.SVM_sigmoid:
            # df_config = pd.read_csv('new_results/svc/best-sigmoid-1.csv')
            df_config = pd.read_csv('new_results\\svc\\best-sigmoid-1.csv')
        elif en == Algorithms.SVM_poly:
            # df_config = pd.read_csv('new_results/svc/best-poly-1.csv')
            df_config = pd.read_csv('new_results\\svc\\best-poly-1.csv')
        elif en == Algorithms.XGB:
            # df_config = pd.read_csv('new_results/xgb/best-xgb-1.csv')
            df_config = pd.read_csv('new_results\\xgb\\best-xgb-1.csv')
        elif en == Algorithms.MLP:
            # df_config = pd.read_csv('new_results/mlp/best-mlp-1.csv')
            df_config = pd.read_csv('new_results\\mlp\\best-mlp-1.csv')
        elif en == Algorithms.GPC:
            # df_config = pd.read_csv('new_results/gpc/best-gpc-1.csv')
            df_config = pd.read_csv('new_results\\gpc\\best-gpc-1.csv')
        else:
            # df_config = pd.read_csv('new_results/gpc/best-gpc-1.csv')
            df_config = pd.read_csv('new_results\\gpc\\best-gpc-1.csv')
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
            classifier = create_SVM_classifier(SVM_Kernels.rbf, row)
        elif (alg == Algorithms.SVM_poly):
            classifier = create_SVM_classifier(SVM_Kernels.poly, row)
        elif (alg == Algorithms.SVM_sigmoid):
            classifier = create_SVM_classifier(SVM_Kernels.sigmoid, row)
        elif (alg == Algorithms.SVM_linear):
            classifier = create_SVM_classifier(SVM_Kernels.linear, row)
        elif (alg == Algorithms.MLP):
            classifier = create_MLP_classifier(row)
        elif (alg == Algorithms.GPC):
            classifier = create_GPC_classifier(row)
        else:
            classifier = None
        classifiers.append(classifier)
    return classifiers


def run_algorithm_ensemble_parallel(q_metrics, filename='', path='', stratify=True, train_size=0.8,
                                    normalize_data=True, scaler='min-max', no_repeats=100, ens: [Algorithms] = None,
                                    ):
    if ens == None or len(ens) < 3:
        return
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_ensembles()

    # my_filename = os.path.join(path, 'new_results\\ens', filename)
    my_filename = os.path.join(path, 'results\\ens', filename)

    # df_configs e dictionar
    df_configs = load_df_configs_for_algs(ens)
    # for alg, df_config in df_configs.items():
    #     print(alg, '->', df_config.head())

    algs = list(df_configs.keys())
    # df_configs_values = df_configs.values()
    df_config_1 = df_configs[algs[0]]
    df_config_2 = df_configs[algs[1]]
    df_config_3 = df_configs[algs[2]]

    for index_1, row_1 in df_config_1.iterrows():
        for index_2, row_2 in df_config_2.iterrows():
            for index_3, row_3 in df_config_3.iterrows():
                for i in range(0, no_repeats):
                    label, ensemble = create_label_for_ensemble(algs, [row_1, row_2, row_3])
                    classifiers = create_classifiers(algs, [row_1, row_2, row_3])
                    run_algorithm_ensemble_configuration(metrics, label, ensemble, X, y, classifiers, len(classifiers),
                                                         train_size=train_size, stratify=stratify)

    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.groupby(['label'], as_index=False).agg(
        {'ensemble': 'first', 'precision': 'mean', 'recall': 'mean', 'f1_score': 'mean', 'roc_auc': 'mean'})
    metrics_df = compute_average_metric(metrics_df)
    metrics_df = compute_roc_auc_score_100(metrics_df)
    metrics_df.sort_values(by=['average_metric'], ascending=False, inplace=True)
    string_results_for_queue = metrics_df.to_csv(index=False, header=False, line_terminator='')
    string_results_for_queue = string_results_for_queue.replace("\n", '')
    q_metrics.put(string_results_for_queue)


def run_algorithm_ensemble(filename='', path='', stratify=True, train_size=0.8,
                           normalize_data=True, scaler='min-max', no_repeats=100, ens: [Algorithms] = None):
    if ens == None or len(ens) < 3:
        return
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_ensembles()

    # my_filename = os.path.join(path, 'new_results\\ens', filename)
    my_filename = os.path.join(path, 'results\\ens', filename)

    # df_configs e dictionar
    df_configs = load_df_configs_for_algs(ens)
    # for alg, df_config in df_configs.items():
    #     print(alg, '->', df_config.head())

    algs = list(df_configs.keys())
    # df_configs_values = df_configs.values()
    df_config_1 = df_configs[algs[0]]
    df_config_2 = df_configs[algs[1]]
    df_config_3 = df_configs[algs[2]]

    for index_1, row_1 in df_config_1.iterrows():
        for index_2, row_2 in df_config_2.iterrows():
            for index_3, row_3 in df_config_3.iterrows():
                for i in range(0, no_repeats):
                    label, ensemble = create_label_for_ensemble(algs, [row_1, row_2, row_3])
                    classifiers = create_classifiers(algs, [row_1, row_2, row_3])

                    run_algorithm_ensemble_configuration(metrics, label, ensemble, X, y, classifiers, len(classifiers),
                                                         train_size=train_size, stratify=stratify)

    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.groupby(['label'], as_index=False).agg(
        {'precision': 'mean', 'recall': 'mean', 'f1_score': 'mean', 'roc_auc': 'mean'})
    metrics_df = compute_average_metric(metrics_df)
    metrics_df = compute_roc_auc_score_100(metrics_df)
    metrics_df.sort_values(by=['average_metric'], ascending=False, inplace=True)
    metrics = appendMetricsTOCSV(my_filename, metrics_df, init_metrics_for_ensembles, header=False)


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
    ensemble = ''
    for alg in algs:
        label += str(alg.value).upper() + " "
        if ensemble == '':
            ensemble += str(alg.value).upper()
        else:
            ensemble += "-" + str(alg.value).upper()

    label += '::'
    for i in range(0, len(algs)):
        alg = algs[i]
        row = rows[i]
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
        label += " :: "
    return label, ensemble
