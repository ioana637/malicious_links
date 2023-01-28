import os
from itertools import chain

import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("error")

from utils import split_data, prediction, cal_metrics, appendMetricsTOCSV


def run_algorithm_SVC_linear_kernel_configuration(metrics, label, X, y,
                                                  tol=1e-4,
                                                  C=1.0,
                                                  shrinking=True,
                                                  cache_size=200,
                                                  class_weight='balanced',
                                                  max_iter=1000,
                                                  decision_function_shape='ovr',
                                                  stratify=False, train_size=0.8,
                                                  normalize_data=False, scaler='min-max'
                                                  ):
    X_train, X_test, y_train, y_test = split_data(X, y, normalize_data=normalize_data, stratify=stratify,
                                                  train_size=train_size, scaler=scaler)
    try:
        # Creating the classifier object
        classifier = SVC(probability=True, kernel='linear', tol=tol, shrinking=shrinking, cache_size=cache_size,
                         C=C, class_weight=class_weight, max_iter=max_iter,
                         decision_function_shape=decision_function_shape)

        # Performing training
        classifier.fit(X_train, y_train)

        # Make predictions
        y_pred, y_pred_probabilities = prediction(X_test, classifier)

        # Compute metrics
        precision, recall, f1, roc_auc = cal_metrics(y_test, y_pred, y_pred_probabilities, label)
        metrics['label'].append(label)

        metrics['tol'].append(tol)
        metrics['C'].append(C)
        metrics['max_iter'].append(max_iter)
        metrics['shrinking'].append(shrinking)
        metrics['cache_size'].append(cache_size)
        metrics['class_weight'].append(class_weight)
        metrics['decision_function_shape'].append(decision_function_shape)

        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1_score'].append(f1)
        metrics['roc_auc'].append(roc_auc)
    except Exception as err:
        pass
        # print(err)
    except RuntimeWarning as warn:
        # print(warn)
        pass


def init_metrics_for_SVM_with_linear_kernel():
    return {'label': [],
            'decision_function_shape': [], 'cache_size': [], 'tol': [], 'C': [], 'shrinking': [],
            'class_weight': [], 'max_iter': [],
            'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': []
            }


def run_algorithm_SVC_linear_kernel(df, filename='', stratify=False, train_size=0.8, normalize_data=False,
                                    scaler='min-max'):
    y = df['label'].copy()
    X = df.drop('label', axis=1).copy()
    metrics = init_metrics_for_SVM_with_linear_kernel()

    path_to_script = os.path.dirname(os.path.abspath(__file__))
    my_filename = os.path.join(path_to_script, 'svc', filename)

    # default algorithm
    run_algorithm_SVC_linear_kernel_configuration(metrics, 'SVC with linear kernel - default params', X, y,
                                                  stratify=stratify, train_size=train_size,
                                                  normalize_data=normalize_data, scaler=scaler
                                                  )
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_SVM_with_linear_kernel, header=True)

    # # CALIBRATING tol
    # for tol in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
    #     run_algorithm_SVC_linear_kernel_configuration(metrics, 'SVC with linear kernel - tol = ' + str(tol), X, y,
    #                                                   tol=tol, stratify=stratify, train_size=train_size,
    #                                                   normalize_data=normalize_data, scaler=scaler)
    #     run_algorithm_SVC_linear_kernel_configuration(metrics, 'SVC with linear kernel - shrinking = False, tol = ' +
    #                                                   str(tol), X, y, tol=tol, shrinking=False, stratify=stratify,
    #                                                   train_size=train_size, normalize_data=normalize_data,
    #                                                   scaler=scaler)
    #
    #     # CALIBRATING C
    # for C in [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 100]:
    #     run_algorithm_SVC_linear_kernel_configuration(metrics, 'SVC with linear kernel - C = ' + str(C), X, y, C=C,
    #                                                   stratify=stratify, train_size=train_size,
    #                                                   normalize_data=normalize_data, scaler=scaler)
    #     run_algorithm_SVC_linear_kernel_configuration(metrics, 'SVC with linear kernel - shrinking = False, C = ' +
    #                                                   str(C), X, y, C=C, shrinking=False, stratify=stratify,
    #                                                   train_size=train_size, normalize_data=normalize_data,
    #                                                   scaler=scaler)
    #
    #     # CALIBRATING class_weight
    # run_algorithm_SVC_linear_kernel_configuration(metrics, 'SVC with linear kernel - class_weight = balanced', X, y,
    #                                               class_weight='balanced', stratify=stratify, train_size=train_size,
    #                                               normalize_data=normalize_data, scaler=scaler)
    # run_algorithm_SVC_linear_kernel_configuration(metrics,
    #                                               'SVC with linear kernel - shrinking = False, class_weight = balanced',
    #                                               X, y, class_weight='balanced', shrinking=False, stratify=stratify,
    #                                               train_size=train_size, normalize_data=normalize_data, scaler=scaler)
    #
    # # CALIBRATING max_iter
    # # for max_iter in [100, 200, 300, 400, 500, 600, 800, 900, 925, 950, 975, 1000, 1025, 1050, 1075, 1100, 1200, 1300,
    # #                  1400, 1500, 1600, 1700, 1800, 1900, 2000]:
    # #     run_algorithm_SVC_linear_kernel_configuration(metrics, 'SVC with linear kernel - max_iter = ' + str(max_iter),
    # #                                                   X, y, max_iter=max_iter, stratify=stratify, train_size=train_size,
    # #                                                   normalize_data=normalize_data, scaler=scaler)
    # #     run_algorithm_SVC_linear_kernel_configuration(metrics,
    # #                                                   'SVC with linear kernel - shrinking = False, max_iter = ' +
    # #                                                   str(max_iter), X, y, shrinking=False, max_iter=max_iter,
    # #                                                   stratify=stratify, train_size=train_size,
    # #                                                   normalize_data=normalize_data, scaler=scaler)
    #
    # # CALIBRATING cache_size
    # for cache_size in [100, 125, 150, 175, 190, 200, 210, 220, 230, 240, 250, 275, 300, 325, 350, 375, 400, 450, 500,
    #                    550, 600, 650, 700, 750, 800]:
    #     run_algorithm_SVC_linear_kernel_configuration(metrics,
    #                                                   'SVC with linear kernel - cache_size = ' + str(cache_size), X, y,
    #                                                   cache_size=cache_size, stratify=stratify, train_size=train_size,
    #                                                   normalize_data=normalize_data, scaler=scaler)
    #     run_algorithm_SVC_linear_kernel_configuration(metrics,
    #                                                   'SVC with linear kernel - shrinking = False, cache_size = ' +
    #                                                   str(cache_size), X, y, shrinking=False, cache_size=cache_size,
    #                                                   stratify=stratify, train_size=train_size,
    #                                                   normalize_data=normalize_data, scaler=scaler)
    # metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_SVM_with_linear_kernel)

    # TODO: our proposed SVC
    for shrinking in [True, False]:
        for C in chain(range(48, 53), np.random.uniform(low=1.0, high=8.0, size=(5,))):
            # 10
            for tol in chain([1e-3], np.random.uniform(low=1e-5, high=1e-10, size=(5,))):
                # 6
                for max_iter in chain([-1], range(190, 210, 2), range(295, 305,2), range(495, 505, 2),
                                      range(595, 605, 2), range(995, 1005, 2), range(1045, 1055, 2),
                                      range(1070, 1080,2),
                                      range(1295, 1305, 2), range(1495, 1505, 2), range(1595, 1605, 2),
                                      range(1695, 1705, 2), range(1795, 1805, 2), range(1895, 1905, 2)):
                    #                     1 + 10 +60 = 71
                    for cache_size in chain([200], range(120, 130, 2), range(170, 180, 2), range(235, 245, 2),
                                            range(370, 380, 2), range(395, 405, 2), range(545, 555, 2),
                                            range(645, 655, 2)):
                        #                         1+70/2 = 36
                        try:
                            run_algorithm_SVC_linear_kernel_configuration(metrics,
                                                                          'SVC with linear kernel - class_weight = balanced, cache_size = ' +
                                                                          str(cache_size) + ', max_iter = ' + str(
                                                                              max_iter) + ', tol = ' +
                                                                          str(tol) + ', C = ' + str(
                                                                              C) + ', shrinking = ' + str(shrinking),
                                                                          X, y, cache_size=cache_size,
                                                                          shrinking=shrinking,
                                                                          C=C, tol=tol, max_iter=max_iter,
                                                                          stratify=stratify, train_size=train_size,
                                                                          normalize_data=normalize_data, scaler=scaler)
                        except Exception as err:
                            print(err)
                    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_SVM_with_linear_kernel)

    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_SVM_with_linear_kernel)


def run_algorithm_SVC_RBF_kernel_configuration(metrics, label, X, y,
                                               tol=1e-4,
                                               C=1.0,
                                               shrinking=True,
                                               cache_size=200,
                                               class_weight='balanced',
                                               max_iter=1000,
                                               gamma='scale',
                                               stratify=False, train_size=0.8,
                                               normalize_data=False, scaler='min-max'
                                               ):
    X_train, X_test, y_train, y_test = split_data(X, y, normalize_data=normalize_data, stratify=stratify,
                                                  train_size=train_size, scaler=scaler)
    try:
        # Creating the classifier object
        classifier = SVC(probability=True, kernel='rbf', tol=tol, shrinking=shrinking, cache_size=cache_size,
                         C=C, class_weight=class_weight, max_iter=max_iter, gamma=gamma)

        # Performing training
        classifier.fit(X_train, y_train)

        # Make predictions
        y_pred, y_pred_probabilities = prediction(X_test, classifier)

        # Compute metrics
        precision, recall, f1, roc_auc = cal_metrics(y_test, y_pred, y_pred_probabilities, label)
        metrics['label'].append(label)

        metrics['tol'].append(tol)
        metrics['C'].append(C)
        metrics['shrinking'].append(shrinking)
        metrics['cache_size'].append(cache_size)
        metrics['class_weight'].append(class_weight)
        metrics['max_iter'].append(max_iter)
        metrics['gamma'].append(gamma)

        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1_score'].append(f1)
        metrics['roc_auc'].append(roc_auc)
    except Exception as err:
        # print(err)
        pass
    except RuntimeWarning as warn:
        # print(warn)
        pass

def init_metrics_for_SVM_with_RBF_kernel():
    return {'label': [],
            'gamma': [], 'cache_size': [],
            'tol': [], 'C': [], 'shrinking': [], 'class_weight': [], 'max_iter': [],
            'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': []
            }


def run_algorithm_SVC_RBF_kernel(df, filename='', stratify=False, train_size=0.8,
                                 normalize_data=False, scaler='min-max'):
    y = df['label'].copy()
    X = df.drop('label', axis=1).copy()
    metrics = init_metrics_for_SVM_with_RBF_kernel()

    path_to_script = os.path.dirname(os.path.abspath(__file__))
    my_filename = os.path.join(path_to_script, 'svc', filename)

    # default algorithm
    run_algorithm_SVC_RBF_kernel_configuration(metrics, 'SVC with RBF kernel - default params', X, y,
                                               stratify=stratify, train_size=train_size,
                                               normalize_data=normalize_data, scaler=scaler)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_SVM_with_RBF_kernel, header=True)
    # CALIBRATING degree, gamma and coef0
    # run_algorithm_SVC_RBF_kernel_configuration(metrics, 'SVC with RBF kernel - gamma = auto', X, y,
    #                                            gamma='auto', stratify=stratify, train_size=train_size,
    #                                            normalize_data=normalize_data, scaler=scaler)
    # for i in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]:
    #     for gamma in [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01,
    #                   0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
    #         run_algorithm_SVC_RBF_kernel_configuration(metrics, 'SVC with RBF kernel - gamma = ' + str(gamma + i), X, y,
    #                                                    gamma=(gamma + i), stratify=stratify, train_size=train_size,
    #                                                    normalize_data=normalize_data, scaler=scaler)
    #
    # # CALIBRATING tol
    # for tol in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
    #     run_algorithm_SVC_RBF_kernel_configuration(metrics, 'SVC with RBF kernel - tol = ' + str(tol), X, y, tol=tol,
    #                                                stratify=stratify, train_size=train_size,
    #                                                normalize_data=normalize_data, scaler=scaler)
    #     run_algorithm_SVC_RBF_kernel_configuration(metrics,
    #                                                'SVC with RBF kernel - shrinking = False, tol = ' + str(tol), X, y,
    #                                                tol=tol, shrinking=False, stratify=stratify, train_size=train_size,
    #                                                normalize_data=normalize_data, scaler=scaler)
    #
    # # CALIBRATING C
    # for C in [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 100]:
    #     run_algorithm_SVC_RBF_kernel_configuration(metrics, 'SVC with RBF kernel - C = ' + str(C), X, y, C=C,
    #                                                stratify=stratify, train_size=train_size,
    #                                                normalize_data=normalize_data, scaler=scaler)
    #     run_algorithm_SVC_RBF_kernel_configuration(metrics, 'SVC with RBF kernel - shrinking = False, C = ' + str(C),
    #                                                X, y, C=C, shrinking=False, stratify=stratify, train_size=train_size,
    #                                                normalize_data=normalize_data, scaler=scaler)
    #
    # # CALIBRATING class_weight
    # run_algorithm_SVC_RBF_kernel_configuration(metrics, 'SVC with RBF kernel - class_weight = balanced', X, y,
    #                                            class_weight='balanced', stratify=stratify, train_size=train_size,
    #                                            normalize_data=normalize_data, scaler=scaler)
    # run_algorithm_SVC_RBF_kernel_configuration(metrics,
    #                                            'SVC with RBF kernel - shrinking = False, class_weight = balanced',
    #                                            X, y, class_weight='balanced', shrinking=False, stratify=stratify,
    #                                            train_size=train_size, normalize_data=normalize_data, scaler=scaler)
    #
    # # CALIBRATING max_iter
    # for max_iter in [100, 200, 300, 400, 500, 600, 800, 900, 925, 950, 975, 1000, 1025, 1050, 1075, 1100, 1200, 1300,
    #                  1400, 1500, 1600, 1700, 1800, 1900, 2000]:
    #     run_algorithm_SVC_RBF_kernel_configuration(metrics, 'SVC with RBF kernel - max_iter = ' + str(max_iter), X, y,
    #                                                max_iter=max_iter, stratify=stratify, train_size=train_size,
    #                                                normalize_data=normalize_data, scaler=scaler)
    #     run_algorithm_SVC_RBF_kernel_configuration(metrics, 'SVC with RBF kernel - shrinking = False, max_iter = ' +
    #                                                str(max_iter), X, y, shrinking=False, max_iter=max_iter,
    #                                                stratify=stratify, train_size=train_size,
    #                                                normalize_data=normalize_data, scaler=scaler)
    #
    # # CALIBRATING cache_size
    # for cache_size in [100, 125, 150, 175, 190, 200, 210, 220, 230, 240, 250, 275, 300, 325, 350, 375, 400, 450, 500,
    #                    550, 600, 650, 700, 750, 800]:
    #     run_algorithm_SVC_RBF_kernel_configuration(metrics, 'SVC with RBF kernel - cache_size = ' + str(cache_size), X,
    #                                                y,
    #                                                cache_size=cache_size, stratify=stratify, train_size=train_size,
    #                                                normalize_data=normalize_data, scaler=scaler)
    #     run_algorithm_SVC_RBF_kernel_configuration(metrics, 'SVC with RBF kernel - shrinking = False, cache_size = ' +
    #                                                str(cache_size), X, y, shrinking=False, cache_size=cache_size,
    #                                                stratify=stratify, train_size=train_size,
    #                                                normalize_data=normalize_data, scaler=scaler)
    # metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_SVM_with_RBF_kernel)

    # our proposed RBF SVC
    for shrinking in [True, False]:
        for max_iter in chain(range(295, 305,2), range(495, 505, 2), range(1045, 1055, 2), range(1395, 1405, 2),
                              range(1605, 1705,2), [-1]):
            # 26
            for cache_size in chain(range(95, 105, 2), range(120, 130, 2), range(145, 155, 2),
                                    range(270, 280, 2), range(295, 305, 2), range(395, 405, 2), [200]):
                # 31
                for gamma in chain(['scale'], np.random.uniform(low=0.0009, high=0.0019, size=(5,))):
                    # 6
                    for C in chain(range(1, 55, 2), range(95, 105, 2),
                                   np.random.uniform(low=1.0, high=1.5, size=(5,))):
                        #                        27 + 10 =37
                        run_algorithm_SVC_RBF_kernel_configuration(metrics, 'SVC with RBF kernel - shrinking = ' + str(
                            shrinking) +
                                                                   ', cache_size = ' + str(
                            cache_size) + ', gamma = ' + str(gamma)
                                                                   + ', C=' + str(C) + ', max_iter=' + str(max_iter), X,
                                                                   y,
                                                                   shrinking=False, cache_size=cache_size, gamma=gamma,
                                                                   max_iter=max_iter, C=C,
                                                                   stratify=stratify, train_size=train_size,
                                                                   normalize_data=normalize_data, scaler=scaler)
                    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_SVM_with_RBF_kernel)

    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_SVM_with_RBF_kernel)


def run_algorithm_SVC_sigmoid_kernel_configuration(metrics, label, X, y,
                                                   tol = 1e-4,
                                                   C=1.0,
                                                   shrinking = True,
                                                   cache_size = 200,
                                                   class_weight = 'balanced',
                                                   max_iter = 1000,
                                                   gamma = 'scale',
                                                   coef0 = 0.0,
                                               stratify=False, train_size=0.8,
                                               normalize_data=False, scaler='min-max'
                                               ):
    X_train, X_test, y_train, y_test = split_data(X, y, normalize_data=normalize_data, stratify=stratify,
                                                  train_size=train_size, scaler=scaler)

    # Creating the classifier object
    classifier = SVC( probability=True, kernel = 'sigmoid', tol = tol,  shrinking=shrinking, cache_size = cache_size,
                  C = C, class_weight = class_weight, max_iter = max_iter, gamma= gamma,
                  coef0 = coef0)

    try:

    # Performing training
        classifier.fit(X_train, y_train)

        # Make predictions
        y_pred, y_pred_probabilities = prediction(X_test, classifier)

        # Compute metrics
        precision, recall, f1, roc_auc = cal_metrics(y_test, y_pred, y_pred_probabilities, label)
        metrics['label'].append(label)

        metrics['tol'].append(tol)
        metrics['C'].append(C)
        metrics['shrinking'].append(shrinking)
        metrics['cache_size'].append(cache_size)
        metrics['class_weight'].append(class_weight)
        metrics['max_iter'].append(max_iter)
        metrics['gamma'].append(gamma)
        metrics['coef0'].append(coef0)

        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1_score'].append(f1)
        metrics['roc_auc'].append(roc_auc)
    except Exception as err:
        # print(err)
        pass
    except RuntimeWarning as warn:
        # print(warn)
        pass

def init_metrics_for_SVM_with_sigmoid_kernel():
    return {'label': [],
            'gamma': [], 'cache_size': [], 'coef0':[],
            'tol': [], 'C': [], 'shrinking': [], 'class_weight': [], 'max_iter': [],
            'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': []
            }


def run_algorithm_SVC_sigmoid_kernel(df, filename='', stratify=False, train_size=0.8,
                                 normalize_data=False, scaler='min-max'):
    y = df['label'].copy()
    X = df.drop('label', axis=1).copy()
    metrics = init_metrics_for_SVM_with_sigmoid_kernel()

    path_to_script = os.path.dirname(os.path.abspath(__file__))
    my_filename = os.path.join(path_to_script, 'svc', filename)

    # default algorithm
    run_algorithm_SVC_sigmoid_kernel_configuration(metrics, 'SVC with sigmoid kernel - default params', X, y,
                                                   stratify = stratify, train_size = train_size,
                                                   normalize_data=normalize_data, scaler=scaler)

    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_SVM_with_sigmoid_kernel, header=True)
    # # CALIBRATING degree, gamma and coef0
    # run_algorithm_SVC_sigmoid_kernel_configuration(metrics, 'SVC with sigmoid kernel - gamma = auto', X, y,
    #                                                gamma = 'auto', stratify = stratify, train_size = train_size,
    #                                                normalize_data=normalize_data, scaler=scaler)
    # for i in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]:
    #     for gamma in [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01,
    #                   0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
    #         run_algorithm_SVC_sigmoid_kernel_configuration(metrics, 'SVC with sigmoid kernel - gamma = ' + str(gamma+ i),
    #                                                        X, y, gamma =( gamma + i), stratify = stratify,
    #                                                        train_size = train_size, normalize_data=normalize_data, scaler=scaler)
    #         for coef0 in [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 30, 40, 50, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]:
    #             run_algorithm_SVC_sigmoid_kernel_configuration(metrics, 'SVC with sigmoid kernel - gamma = ' + str(gamma+ i)
    #                                                            +', coef0 = ' + str(coef0), X, y,coef0 = coef0,
    #                                                            gamma =( gamma + i), stratify = stratify, train_size = train_size,
    #                                                            normalize_data=normalize_data, scaler=scaler)
    #
    #
    # # CALIBRATING tol
    # for tol in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
    #     run_algorithm_SVC_sigmoid_kernel_configuration(metrics, 'SVC with sigmoid kernel - tol = '+ str(tol), X, y,
    #                                                    tol = tol, stratify = stratify, train_size = train_size,
    #                                                    normalize_data=normalize_data, scaler=scaler)
    #     run_algorithm_SVC_sigmoid_kernel_configuration(metrics, 'SVC with sigmoid kernel - shrinking = False, tol = '+
    #                                                    str(tol), X, y, tol = tol, shrinking=False,stratify = stratify,
    #                                                    train_size = train_size, normalize_data=normalize_data, scaler=scaler)
    #
    # # CALIBRATING C
    # for C in [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 100]:
    #     run_algorithm_SVC_sigmoid_kernel_configuration(metrics, 'SVC with sigmoid kernel - C = '+ str(C), X, y,
    #                                                    C = C, stratify = stratify, train_size = train_size,
    #                                                    normalize_data=normalize_data, scaler=scaler)
    #     run_algorithm_SVC_sigmoid_kernel_configuration(metrics, 'SVC with sigmoid kernel - shrinking = False, C = '+
    #                                                    str(C), X, y, C = C, shrinking=False,stratify = stratify,
    #                                                    train_size = train_size, normalize_data=normalize_data, scaler=scaler)
    #
    # # CALIBRATING class_weight
    # run_algorithm_SVC_sigmoid_kernel_configuration(metrics, 'SVC with sigmoid kernel - class_weight = balanced',
    #                                                X, y, class_weight='balanced', stratify = stratify, train_size = train_size,
    #                                                normalize_data=normalize_data, scaler=scaler)
    # run_algorithm_SVC_sigmoid_kernel_configuration(metrics, 'SVC with sigmoid kernel - shrinking = False, class_weight = balanced',
    #                                                X, y, class_weight = 'balanced', shrinking=False,stratify = stratify,
    #                                                train_size = train_size, normalize_data=normalize_data, scaler=scaler)
    #
    # # CALIBRATING max_iter
    # for max_iter in [100, 200, 300, 400, 500, 600, 800, 900, 925, 950, 975, 1000, 1025, 1050, 1075, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]:
    #     run_algorithm_SVC_sigmoid_kernel_configuration(metrics, 'SVC with sigmoid kernel - max_iter = '+ str(max_iter),
    #                                                    X, y, max_iter = max_iter, stratify = stratify, train_size = train_size,
    #                                                    normalize_data=normalize_data, scaler=scaler)
    #     run_algorithm_SVC_sigmoid_kernel_configuration(metrics, 'SVC with sigmoid kernel - shrinking = False, max_iter = '+
    #                                                    str(max_iter), X, y, shrinking = False, max_iter = max_iter,
    #                                                    stratify = stratify, train_size = train_size,
    #                                                    normalize_data=normalize_data, scaler=scaler)
    #
    # # CALIBRATING cache_size
    # for cache_size in [100, 125, 150, 175, 190, 200, 210, 220, 230, 240, 250, 275, 300, 325, 350, 375, 400, 450, 500, 550, 600, 650, 700, 750, 800]:
    #     run_algorithm_SVC_sigmoid_kernel_configuration(metrics, 'SVC with sigmoid kernel - cache_size = '+ str(cache_size),
    #                                                    X, y, cache_size = cache_size, stratify = stratify, train_size = train_size,
    #                                                    normalize_data=normalize_data, scaler=scaler)
    #     run_algorithm_SVC_sigmoid_kernel_configuration(metrics, 'SVC with sigmoid kernel - shrinking = False, cache_size = '
    #                                                    + str(cache_size), X, y, shrinking = False, cache_size = cache_size,
    #                                                    stratify = stratify, train_size = train_size,
    #                                                    normalize_data=normalize_data, scaler=scaler)
    #
    # metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_SVM_with_sigmoid_kernel)

    for shrinking in [True, False]:
        for cache_size in chain(range(145, 155, 2), range(345, 355, 2)):
            # 10
            for coef0 in chain([0.0], range(5, 15,2), range(35, 45, 2), range(195, 205, 2), range(395, 405, 2),
                               range(695, 705, 2), range(795, 805, 2), range(895, 905, 2),
                               np.random.uniform(low=0.0, high=0.8, size=(20,)),
                               np.random.uniform(low=1.0, high=3.0, size=(10,)),
                               np.random.uniform(low=4.5, high=5.5, size=(10,)),
                               np.random.uniform(low=7.0, high=8.0, size=(10,)),
                               np.random.uniform(low=12.0, high=13.0, size=(10,))):
                # 36+60=96
                for gamma in chain(['scale'], np.random.uniform(low=0.001, high=0.009, size=(10,)),
                                   np.random.uniform(low=0.01, high=0.09, size=(10,)),
                                   np.random.uniform(low=0.1, high=0.2, size=(10,)),
                                   np.random.uniform(low=0.85, high=0.95, size=(10,))):
                    # 41
                    run_algorithm_SVC_sigmoid_kernel_configuration(metrics, 'SVC with sigmoid kernel - shrinking = '+str(shrinking)+
                                                                   ', cache_size = '+ str(cache_size)+', gamma = '+
                                                                   str(gamma)+', coef0='+str(coef0), X, y,
                                                                   shrinking = False, cache_size = cache_size,
                                                                   gamma=gamma, coef0=coef0,
                                                                   stratify = stratify, train_size = train_size,
                                                                   normalize_data=normalize_data, scaler=scaler)
                metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_SVM_with_sigmoid_kernel)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_SVM_with_sigmoid_kernel)


