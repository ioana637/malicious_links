import os
from itertools import chain
from multiprocessing import Pool, Manager

import numpy as np
from sklearn.svm import SVC
import warnings

from data_pre import split_data_in_testing_training, load_normalized_dataset

warnings.filterwarnings("error")

from utils import prediction, cal_metrics, appendMetricsTOCSV, convert_metrics_to_csv, listener_write_to_file


def run_algorithm_SVC_linear_kernel_configuration_parallel(X, y, q_metrics,
                                                           tol=1e-4, C=1.0, shrinking=True, cache_size=200,
                                                           class_weight='balanced', max_iter=1000,
                                                           decision_function_shape='ovr',
                                                           stratify=False, train_size=0.8):
    X_test, X_train, y_test, y_train = split_data_in_testing_training(X, y, stratify, train_size)
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
        precision, recall, f1, roc_auc = cal_metrics(y_test, y_pred, y_pred_probabilities)
        label = 'SVM with linear kernel'
        string_results_for_queue = convert_metrics_to_csv(',', label, tol, C, max_iter, shrinking, cache_size,
                                                          class_weight,
                                                          decision_function_shape, precision, recall, f1, roc_auc)

        q_metrics.put(string_results_for_queue)
    except Exception as err:
        # pass
        print(err)
    except RuntimeWarning as warn:
        print(warn)
        # pass


def run_algorithm_SVC_linear_kernel_configuration(metrics, label, X, y,
                                                  tol=1e-4,
                                                  C=1.0,
                                                  shrinking=True,
                                                  cache_size=200,
                                                  class_weight='balanced',
                                                  max_iter=1000,
                                                  decision_function_shape='ovr',
                                                  stratify=False, train_size=0.8):
    X_test, X_train, y_test, y_train = split_data_in_testing_training(X, y, stratify, train_size)
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
        precision, recall, f1, roc_auc = cal_metrics(y_test, y_pred, y_pred_probabilities)
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


def run_algorithm_SVC_linear_kernel_parallel(filename='', path='', stratify=False, train_size=0.8,
                                             normalize_data=False, scaler='min-max'):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_SVM_with_linear_kernel()
    my_filename = os.path.join(path, 'results/svc', filename)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_SVM_with_linear_kernel, header=True)
    max_iter_list = list(chain([-1], range(450, 550, 10), range(950, 1250, 10), range(1350, 1450, 10),
                               range(1550, 1650, 10), range(1950, 2050, 10)))
    shrinking_list = [True, False]
    C_list = list(chain([1, 1.4, 1.5, 1.6, 1.7, 1.8, 4], np.random.uniform(low=1.0, high=4.5, size=(10,))))
    tol_list = list(chain([1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, ], np.random.uniform(low=1e-2, high=1e-9, size=(10,))))
    cache_size_list = list(chain(range(195, 205, 2), range(445, 455, 2), range(645, 655, 2), range(695, 705, 2)))

    # count = len(max_iter_list) * len(i_list) * len(j_list) * len(k_list) * len(n_iter_no_change_list) * \
    #         len(epsilon_list) * len(learning_rate_init_list) * len(solver_list) * len(activation_list)

    with Manager() as manager:
        q_metrics = manager.Queue()
        jobs = []

        with Pool(14) as pool:
            watcher = pool.apply_async(listener_write_to_file, (q_metrics, my_filename))
            for shrinking in shrinking_list:
                for C in C_list:
                    for tol in tol_list:
                        for max_iter in max_iter_list:
                            for cache_size in cache_size_list:
                                job = pool.apply_async(run_algorithm_SVC_linear_kernel_configuration_parallel,
                                                       (X, y, q_metrics,
                                                        tol, C, shrinking, cache_size, 'balanced', max_iter, 'ovr',
                                                        stratify, train_size))
                                # print(job)
                                jobs.append(job)
            # print(jobs)
            # collect results from the workers through the pool result queue
            for job in jobs:
                job.get()
            # now we are done, kill the listener
            q_metrics.put('kill')
            pool.close()
            pool.join()


def run_algorithm_SVC_linear_kernel(filename='', path='', stratify=False, train_size=0.8,
                                    normalize_data=False, scaler='min-max'):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_SVM_with_linear_kernel()

    # path_to_script = os.path.dirname(os.path.abspath(__file__))
    my_filename = os.path.join(path, 'results/svm', filename)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_SVM_with_linear_kernel, header=True)

    # default algorithm
    # run_algorithm_SVC_linear_kernel_configuration(metrics, 'SVC with linear kernel - default params', X, y,
    #                                               stratify=stratify, train_size=train_size
    #                                               )
    # metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_SVM_with_linear_kernel, header=True)
    #
    # # CALIBRATING tol
    # for tol in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
    #     run_algorithm_SVC_linear_kernel_configuration(metrics, 'SVC with linear kernel - tol = ' + str(tol), X, y,
    #                                                   tol=tol, class_weight='balanced', stratify=stratify,
    #                                                   train_size=train_size)
    #     run_algorithm_SVC_linear_kernel_configuration(metrics, 'SVC with linear kernel - shrinking = False, tol = ' +
    #                                                   str(tol), X, y, tol=tol, shrinking=False, class_weight='balanced',
    #                                                   stratify=stratify,
    #                                                   train_size=train_size)
    #
    #     # CALIBRATING C
    # for C in [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 100]:
    #     run_algorithm_SVC_linear_kernel_configuration(metrics, 'SVC with linear kernel - C = ' + str(C), X, y, C=C,
    #                                                   class_weight='balanced',
    #                                                   stratify=stratify, train_size=train_size)
    #     run_algorithm_SVC_linear_kernel_configuration(metrics, 'SVC with linear kernel - shrinking = False, C = ' +
    #                                                   str(C), X, y, C=C, shrinking=False, class_weight='balanced',
    #                                                   stratify=stratify,
    #                                                   train_size=train_size)
    #
    #     # CALIBRATING class_weight
    # run_algorithm_SVC_linear_kernel_configuration(metrics, 'SVC with linear kernel - class_weight = balanced', X, y,
    #                                               class_weight='balanced', stratify=stratify, train_size=train_size)
    # run_algorithm_SVC_linear_kernel_configuration(metrics,
    #                                               'SVC with linear kernel - shrinking = False, class_weight = balanced',
    #                                               X, y, class_weight='balanced', shrinking=False, stratify=stratify,
    #                                               train_size=train_size)
    #
    # # CALIBRATING
    # # max_iter
    # for max_iter in [100, 200, 300, 400, 500, 600, 800, 900, 925, 950, 975, 1000, 1025, 1050, 1075, 1100, 1200, 1300,
    #                  1400, 1500, 1600, 1700, 1800, 1900, 2000]:
    #     run_algorithm_SVC_linear_kernel_configuration(metrics, 'SVC with linear kernel - max_iter = ' + str(max_iter),
    #                                                   X, y, max_iter=max_iter, class_weight='balanced',
    #                                                   stratify=stratify, train_size=train_size)
    #     run_algorithm_SVC_linear_kernel_configuration(metrics,
    #                                                   'SVC with linear kernel - shrinking = False, max_iter = ' +
    #                                                   str(max_iter), X, y, shrinking=False, max_iter=max_iter,
    #                                                   class_weight='balanced',
    #                                                   stratify=stratify, train_size=train_size)
    #
    # # CALIBRATING cache_size
    # for cache_size in [100, 125, 150, 175, 190, 200, 210, 220, 230, 240, 250, 275, 300, 325, 350, 375, 400, 450, 500,
    #                    550, 600, 650, 700, 750, 800]:
    #     run_algorithm_SVC_linear_kernel_configuration(metrics,
    #                                                   'SVC with linear kernel - cache_size = ' + str(cache_size), X, y,
    #                                                   cache_size=cache_size, class_weight='balanced', stratify=stratify,
    #                                                   train_size=train_size)
    #     run_algorithm_SVC_linear_kernel_configuration(metrics,
    #                                                   'SVC with linear kernel - shrinking = False, cache_size = ' +
    #                                                   str(cache_size), X, y, shrinking=False, cache_size=cache_size,
    #                                                   class_weight='balanced', stratify=stratify, train_size=train_size)
    # metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_SVM_with_linear_kernel)

    # TODO: our proposed SVC
    for shrinking in [True, False]:
        for C in chain([1, 1.4, 1.5, 1.6, 1.7, 1.8, 4], np.random.uniform(low=1.0, high=4.5, size=(20,))):
            for tol in chain([1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, ],
                             np.random.uniform(low=1e-2, high=1e-9, size=(20,))):
                for max_iter in chain([-1], range(450, 550, 2), range(950, 1250, 2), range(1350, 1450, 2),
                                      range(1550, 1650, 2), range(1950, 2050, 2)):
                    for cache_size in chain(range(195, 205, 1), range(445, 455, 1), range(645, 655, 1),
                                            range(695, 705, 1)):
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
                                                                          class_weight='balanced',
                                                                          decision_function_shape='ovr',
                                                                          stratify=stratify, train_size=train_size)
                        except Exception as err:
                            print(err)
                    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_SVM_with_linear_kernel)

    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_SVM_with_linear_kernel)


def run_algorithm_SVC_RBF_kernel_configuration_parallel(X, y, q_metrics, tol=1e-4, C=1.0, shrinking=True,
                                                        cache_size=200,
                                                        class_weight='balanced', max_iter=1000, gamma='scale',
                                                        stratify=False,
                                                        train_size=0.8):
    X_test, X_train, y_test, y_train = split_data_in_testing_training(X, y, stratify, train_size)
    try:
        # Creating the classifier object
        classifier = SVC(probability=True, kernel='rbf', tol=tol, shrinking=shrinking, cache_size=cache_size,
                         C=C, class_weight=class_weight, max_iter=max_iter, gamma=gamma)
        # Performing training
        classifier.fit(X_train, y_train)
        # Make predictions
        y_pred, y_pred_probabilities = prediction(X_test, classifier)
        label = 'SVM with RBF kernel'
        # Compute metrics
        precision, recall, f1, roc_auc = cal_metrics(y_test, y_pred, y_pred_probabilities, label)
        queue_results_to_csv = convert_metrics_to_csv(',', label, tol, C, shrinking, cache_size, class_weight, max_iter,
                                                      gamma, precision, recall, f1, roc_auc)
        q_metrics.put(queue_results_to_csv)
    except Exception as err:
        print(err)
        # pass
    except RuntimeWarning as warn:
        print(warn)
        # pass


def run_algorithm_SVC_RBF_kernel_configuration(metrics, label, X, y,
                                               tol=1e-4,
                                               C=1.0,
                                               shrinking=True,
                                               cache_size=200,
                                               class_weight='balanced',
                                               max_iter=1000,
                                               gamma='scale',
                                               stratify=False, train_size=0.8
                                               ):
    X_test, X_train, y_test, y_train = split_data_in_testing_training(X, y, stratify, train_size)
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


def run_algorithm_SVC_RBF_kernel_parallel(filename='', path='', stratify=False, train_size=0.8,
                                          normalize_data=False, scaler='min-max'):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_SVM_with_RBF_kernel()
    my_filename = os.path.join(path, 'results/svc', filename)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_SVM_with_RBF_kernel, header=True)
    max_iter_list = list(chain(range(295, 305, 2), range(795, 805, 2), range(895, 905, 2),
                               range(995, 1005, 2), range(1020, 1030, 2), range(1295, 1305, 2), range(1495, 1505, 2),
                               [-1]))
    shrinking_list = [True, False]
    gamma_list = list(chain(['scale'], [0.890, 0.892, 0.894, 0.896, 0.898, 0.9, 0.901, 0.902, 0.903,
                                        0.904, 0.905, 0.906, 0.907, 0.908, 0.909, 0.91, 0.92, 0.93, 0.94, 0.95,
                                        0.96, 0.97, 0.98, 0.99]))
    C_list = list(chain(range(5, 45, 2), np.random.uniform(low=1.0, high=2.5, size=(10,))))
    cache_size_list = list(chain(range(185, 215, 5), range(235, 245, 2), range(445, 455, 2)))

    # count = len(max_iter_list) * len(i_list) * len(j_list) * len(k_list) * len(n_iter_no_change_list) * \
    #         len(epsilon_list) * len(learning_rate_init_list) * len(solver_list) * len(activation_list)

    with Manager() as manager:
        q_metrics = manager.Queue()
        jobs = []

        with Pool(14) as pool:
            watcher = pool.apply_async(listener_write_to_file, (q_metrics, my_filename))
            for shrinking in shrinking_list:
                for max_iter in max_iter_list:
                    for cache_size in cache_size_list:
                        for gamma in gamma_list:
                            for C in C_list:
                                job = pool.apply_async(run_algorithm_SVC_RBF_kernel_configuration_parallel,
                                                       (X, y, q_metrics, 1e-4, C, shrinking, cache_size, 'balanced',
                                                        max_iter, gamma, stratify, train_size))
                                # print(job)
                                jobs.append(job)
            # print(jobs)
            # collect results from the workers through the pool result queue
            for job in jobs:
                job.get()
            # now we are done, kill the listener
            q_metrics.put('kill')
            pool.close()
            pool.join()


def run_algorithm_SVC_RBF_kernel(filename='', path='', stratify=False, train_size=0.8,
                                 normalize_data=False, scaler='min-max'):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_SVM_with_RBF_kernel()

    # path_to_script = os.path.dirname(os.path.abspath(__file__))
    my_filename = os.path.join(path, 'results/svm', filename)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_SVM_with_RBF_kernel, header=True)

    # default algorithm
    # run_algorithm_SVC_RBF_kernel_configuration(metrics, 'SVC with RBF kernel - default params', X, y,
    #                                            stratify=stratify, train_size=train_size)
    # metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_SVM_with_RBF_kernel)
    # # CALIBRATING degree, gamma and coef0
    # run_algorithm_SVC_RBF_kernel_configuration(metrics, 'SVC with RBF kernel - gamma = auto', X, y,
    #                                            gamma='auto', class_weight='balanced', stratify=stratify,
    #                                            train_size=train_size)
    # for i in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]:
    #     for gamma in [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01,
    #                   0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
    #         run_algorithm_SVC_RBF_kernel_configuration(metrics, 'SVC with RBF kernel - gamma = ' + str(gamma + i), X, y,
    #                                                    gamma=(gamma + i), class_weight='balanced', stratify=stratify,
    #                                                    train_size=train_size)
    #
    # # CALIBRATING tol
    # for tol in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
    #     run_algorithm_SVC_RBF_kernel_configuration(metrics, 'SVC with RBF kernel - tol = ' + str(tol), X, y, tol=tol,
    #                                                class_weight='balanced',
    #                                                stratify=stratify, train_size=train_size)
    #     run_algorithm_SVC_RBF_kernel_configuration(metrics,
    #                                                'SVC with RBF kernel - shrinking = False, tol = ' + str(tol), X, y,
    #                                                tol=tol, shrinking=False, class_weight='balanced', stratify=stratify,
    #                                                train_size=train_size)
    #
    # # CALIBRATING C
    # for C in [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 100]:
    #     run_algorithm_SVC_RBF_kernel_configuration(metrics, 'SVC with RBF kernel - C = ' + str(C), X, y, C=C,
    #                                                class_weight='balanced', stratify=stratify, train_size=train_size)
    #     run_algorithm_SVC_RBF_kernel_configuration(metrics, 'SVC with RBF kernel - shrinking = False, C = ' + str(C),
    #                                                X, y, C=C, shrinking=False, class_weight='balanced',
    #                                                stratify=stratify, train_size=train_size)
    #
    # # CALIBRATING class_weight
    # run_algorithm_SVC_RBF_kernel_configuration(metrics, 'SVC with RBF kernel - class_weight = balanced', X, y,
    #                                            class_weight='balanced', stratify=stratify, train_size=train_size,
    #                                            )
    # run_algorithm_SVC_RBF_kernel_configuration(metrics,
    #                                            'SVC with RBF kernel - shrinking = False, class_weight = balanced',
    #                                            X, y, class_weight='balanced', shrinking=False, stratify=stratify,
    #                                            train_size=train_size)
    #
    # # CALIBRATING max_iter
    # for max_iter in [100, 200, 300, 400, 500, 600, 800, 900, 925, 950, 975, 1000, 1025, 1050, 1075, 1100, 1200, 1300,
    #                  1400, 1500, 1600, 1700, 1800, 1900, 2000]:
    #     run_algorithm_SVC_RBF_kernel_configuration(metrics, 'SVC with RBF kernel - max_iter = ' + str(max_iter), X, y,
    #                                                max_iter=max_iter, class_weight='balanced', stratify=stratify,
    #                                                train_size=train_size)
    #     run_algorithm_SVC_RBF_kernel_configuration(metrics, 'SVC with RBF kernel - shrinking = False, max_iter = ' +
    #                                                str(max_iter), X, y, shrinking=False, max_iter=max_iter,
    #                                                class_weight='balanced',
    #                                                stratify=stratify, train_size=train_size)
    #
    # # CALIBRATING cache_size
    # for cache_size in [100, 125, 150, 175, 190, 200, 210, 220, 230, 240, 250, 275, 300, 325, 350, 375, 400, 450, 500,
    #                    550, 600, 650, 700, 750, 800]:
    #     run_algorithm_SVC_RBF_kernel_configuration(metrics, 'SVC with RBF kernel - cache_size = ' + str(cache_size), X,
    #                                                y,
    #                                                cache_size=cache_size, class_weight='balanced', stratify=stratify,
    #                                                train_size=train_size)
    #     run_algorithm_SVC_RBF_kernel_configuration(metrics, 'SVC with RBF kernel - shrinking = False, cache_size = ' +
    #                                                str(cache_size), X, y, shrinking=False, cache_size=cache_size,
    #                                                class_weight='balanced',
    #                                                stratify=stratify, train_size=train_size)
    # metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_SVM_with_RBF_kernel)

    # our proposed RBF SVC
    for shrinking in [True, False]:
        for max_iter in chain(range(295, 305, 1), range(795, 805, 1), range(895, 905, 1),
                              range(995, 1005, 1), range(1020, 1030, 1), range(1295, 1305), range(1495, 1505),
                              [-1]):
            for cache_size in chain(range(185, 215), range(235, 245), range(445, 455)):
                for gamma in chain(['scale'], [0.890, 0.892, 0.894, 0.896, 0.898, 0.9, 0.901, 0.902, 0.903,
                                               0.904, 0.905, 0.906, 0.907, 0.908, 0.909, 0.91, 0.92, 0.93, 0.94, 0.95,
                                               0.96, 0.97, 0.98, 0.99]):
                    for C in chain(range(5, 45, 1), np.random.uniform(low=1.0, high=2.5, size=(10,))):
                        run_algorithm_SVC_RBF_kernel_configuration(metrics, 'SVC with RBF kernel - shrinking = ' + str(
                            shrinking) + ', cache_size = ' + str(cache_size) + ', gamma = ' + str(gamma) + ', C=' +
                                                                   str(C) + ', max_iter=' + str(max_iter), X, y,
                                                                   shrinking=False, cache_size=cache_size, gamma=gamma,
                                                                   max_iter=max_iter, C=C, class_weight='balanced',
                                                                   stratify=stratify, train_size=train_size)
                    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_SVM_with_RBF_kernel)

    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_SVM_with_RBF_kernel)


def run_algorithm_SVC_sigmoid_kernel_configuration(metrics, label, X, y,
                                                   tol=1e-4,
                                                   C=1.0,
                                                   shrinking=True,
                                                   cache_size=200,
                                                   class_weight='balanced',
                                                   max_iter=1000,
                                                   gamma='scale',
                                                   coef0=0.0,
                                                   stratify=False, train_size=0.8
                                                   ):
    X_test, X_train, y_test, y_train = split_data_in_testing_training(X, y, stratify, train_size)

    # Creating the classifier object
    classifier = SVC(probability=True, kernel='sigmoid', tol=tol, shrinking=shrinking, cache_size=cache_size,
                     C=C, class_weight=class_weight, max_iter=max_iter, gamma=gamma,
                     coef0=coef0)

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
            'gamma': [], 'cache_size': [], 'coef0': [],
            'tol': [], 'C': [], 'shrinking': [], 'class_weight': [], 'max_iter': [],
            'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': []
            }


def run_algorithm_SVC_sigmoid_kernel(filename='', path='', stratify=False, train_size=0.8,
                                     normalize_data=False, scaler='min-max'):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_SVM_with_sigmoid_kernel()

    # path_to_script = os.path.dirname(os.path.abspath(__file__))
    my_filename = os.path.join(path, 'results/svm', filename)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_SVM_with_sigmoid_kernel, header=True)

    # default algorithm
    # run_algorithm_SVC_sigmoid_kernel_configuration(metrics, 'SVC with sigmoid kernel - default params', X, y,
    #                                                stratify=stratify, train_size=train_size)
    #
    # # CALIBRATING degree, gamma and coef0
    # run_algorithm_SVC_sigmoid_kernel_configuration(metrics, 'SVC with sigmoid kernel - gamma = auto', X, y,
    #                                                gamma='auto', class_weight='balanced', stratify=stratify,
    #                                                train_size=train_size)
    # for i in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]:
    #     for gamma in [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01,
    #                   0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
    #         run_algorithm_SVC_sigmoid_kernel_configuration(metrics,
    #                                                        'SVC with sigmoid kernel - gamma = ' + str(gamma + i),
    #                                                        X, y, gamma=(gamma + i), class_weight='balanced',
    #                                                        stratify=stratify,
    #                                                        train_size=train_size)
    #         for coef0 in [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5,
    #                       10, 12.5, 15, 17.5, 20, 30, 40, 50, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500,
    #                       600, 700, 800, 900, 1000]:
    #             run_algorithm_SVC_sigmoid_kernel_configuration(metrics,
    #                                                            'SVC with sigmoid kernel - gamma = ' + str(gamma + i)
    #                                                            + ', coef0 = ' + str(coef0), X, y, coef0=coef0,
    #                                                            gamma=(gamma + i), class_weight='balanced',
    #                                                            stratify=stratify,
    #                                                            train_size=train_size)
    #
    # # CALIBRATING tol
    # for tol in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
    #     run_algorithm_SVC_sigmoid_kernel_configuration(metrics, 'SVC with sigmoid kernel - tol = ' + str(tol), X, y,
    #                                                    tol=tol, class_weight='balanced', stratify=stratify,
    #                                                    train_size=train_size)
    #     run_algorithm_SVC_sigmoid_kernel_configuration(metrics, 'SVC with sigmoid kernel - shrinking = False, tol = ' +
    #                                                    str(tol), X, y, tol=tol, shrinking=False,
    #                                                    class_weight='balanced', stratify=stratify,
    #                                                    train_size=train_size)
    #
    # # CALIBRATING C
    # for C in [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 100]:
    #     run_algorithm_SVC_sigmoid_kernel_configuration(metrics, 'SVC with sigmoid kernel - C = ' + str(C), X, y,
    #                                                    C=C, class_weight='balanced', stratify=stratify,
    #                                                    train_size=train_size)
    #     run_algorithm_SVC_sigmoid_kernel_configuration(metrics, 'SVC with sigmoid kernel - shrinking = False, C = ' +
    #                                                    str(C), X, y, C=C, shrinking=False, class_weight='balanced',
    #                                                    stratify=stratify,
    #                                                    train_size=train_size)
    #
    # # CALIBRATING class_weight
    # run_algorithm_SVC_sigmoid_kernel_configuration(metrics, 'SVC with sigmoid kernel - class_weight = balanced',
    #                                                X, y, class_weight='balanced', stratify=stratify,
    #                                                train_size=train_size)
    # run_algorithm_SVC_sigmoid_kernel_configuration(metrics,
    #                                                'SVC with sigmoid kernel - shrinking = False, class_weight = balanced',
    #                                                X, y, class_weight='balanced', shrinking=False, stratify=stratify,
    #                                                train_size=train_size)
    #
    # # CALIBRATING max_iter
    # for max_iter in [100, 200, 300, 400, 500, 600, 800, 900, 925, 950, 975, 1000, 1025, 1050, 1075, 1100, 1200, 1300,
    #                  1400, 1500, 1600, 1700, 1800, 1900, 2000]:
    #     run_algorithm_SVC_sigmoid_kernel_configuration(metrics, 'SVC with sigmoid kernel - max_iter = ' + str(max_iter),
    #                                                    X, y, max_iter=max_iter, class_weight='balanced',
    #                                                    stratify=stratify,
    #                                                    train_size=train_size)
    #     run_algorithm_SVC_sigmoid_kernel_configuration(metrics,
    #                                                    'SVC with sigmoid kernel - shrinking = False, max_iter = ' +
    #                                                    str(max_iter), X, y, shrinking=False, max_iter=max_iter,
    #                                                    class_weight='balanced',
    #                                                    stratify=stratify, train_size=train_size)
    #
    # # CALIBRATING cache_size
    # for cache_size in [100, 125, 150, 175, 190, 200, 210, 220, 230, 240, 250, 275, 300, 325, 350, 375, 400, 450, 500,
    #                    550, 600, 650, 700, 750, 800]:
    #     run_algorithm_SVC_sigmoid_kernel_configuration(metrics,
    #                                                    'SVC with sigmoid kernel - cache_size = ' + str(cache_size),
    #                                                    X, y, cache_size=cache_size, class_weight='balanced',
    #                                                    stratify=stratify,
    #                                                    train_size=train_size)
    #     run_algorithm_SVC_sigmoid_kernel_configuration(metrics,
    #                                                    'SVC with sigmoid kernel - shrinking = False, cache_size = '
    #                                                    + str(cache_size), X, y, shrinking=False, cache_size=cache_size,
    #                                                    class_weight='balanced',
    #                                                    stratify=stratify, train_size=train_size)
    #
    # metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_SVM_with_sigmoid_kernel)

    for shrinking in [True, False]:
        for cache_size in chain(range(195, 205, 1)):
            for coef0 in chain([0.0], range(1, 15, 1),
                               np.random.uniform(low=0.001, high=1.0, size=(20,))):
                for gamma in chain(['scale'], np.random.uniform(low=0.001, high=0.3, size=(50,))):
                    run_algorithm_SVC_sigmoid_kernel_configuration(metrics,
                                                                   'SVC with sigmoid kernel - shrinking = ' + str(
                                                                       shrinking) +
                                                                   ', cache_size = ' + str(cache_size) + ', gamma = ' +
                                                                   str(gamma) + ', coef0=' + str(coef0), X, y,
                                                                   shrinking=False, cache_size=cache_size,
                                                                   gamma=gamma, coef0=coef0, class_weight='balanced',
                                                                   stratify=stratify, train_size=train_size,
                                                                   )
                metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_SVM_with_sigmoid_kernel)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_SVM_with_sigmoid_kernel)


def init_metrics_for_SVM_with_polynomial_kernel():
    return {'label': [],
            'gamma': [], 'cache_size': [], 'coef0': [], 'degree': [],
            'tol': [], 'C': [], 'shrinking': [], 'class_weight': [], 'max_iter': [],
            'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': []
            }


def run_algorithm_SVC_poly_kernel_configuration(metrics, label, X, y,
                                                tol=1e-4,
                                                C=1.0,
                                                shrinking=True,
                                                cache_size=200,
                                                class_weight=None,
                                                max_iter=1000,
                                                gamma='scale',
                                                degree=3,
                                                coef0=0.0,
                                                stratify=False, train_size=0.8
                                                ):
    X_test, X_train, y_test, y_train = split_data_in_testing_training(X, y, stratify, train_size)

    # Creating the classifier object
    classifier = SVC(probability=True, kernel='poly', tol=tol, shrinking=shrinking, cache_size=cache_size,
                     C=C, class_weight=class_weight, max_iter=max_iter,
                     degree=degree, gamma=gamma, coef0=coef0)

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
        metrics['degree'].append(degree)

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


def run_algorithm_SVC_poly_kernel(filename='', path='', stratify=False, train_size=0.8,
                                  normalize_data=False, scaler='min-max'):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_SVM_with_polynomial_kernel()

    # path_to_script = os.path.dirname(os.path.abspath(__file__))
    my_filename = os.path.join(path, 'results/svm', filename)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_SVM_with_polynomial_kernel, header=True)

    # default algorithm
    # run_algorithm_SVC_poly_kernel_configuration(metrics, 'SVC with polynomial kernel - default params', X, y,
    #                                             stratify=stratify, train_size=train_size)
    # metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_SVM_with_polynomial_kernel)
    #
    # # CALIBRATING degree, gamma and coef0
    # for degree in range(1, 20):
    #     run_algorithm_SVC_poly_kernel_configuration(metrics, 'SVC with polynomial kernel - degree = ' + str(degree), X,
    #                                                 y, degree=degree, class_weight='balanced', stratify=stratify,
    #                                                 train_size=train_size)
    #     run_algorithm_SVC_poly_kernel_configuration(metrics, 'SVC with polynomial kernel - degree = ' + str(
    #         degree) + ', gamma = auto', X, y, degree=degree, gamma='auto', class_weight='balanced', stratify=stratify,
    #                                                 train_size=train_size)
    #     for i in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]:
    #         for gamma in [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01,
    #                       0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
    #             run_algorithm_SVC_poly_kernel_configuration(metrics, 'SVC with polynomial kernel - degree = ' + str(
    #                 degree) + ', gamma = ' + str(gamma + i), X, y, degree=degree, gamma=(gamma + i),
    #                                                         class_weight='balanced',
    #                                                         stratify=stratify,
    #                                                         train_size=train_size)
    #             for coef0 in [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5,
    #                           7.5, 10, 12.5, 15, 17.5, 20, 30, 40, 50, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450,
    #                           500, 600, 700, 800, 900, 1000]:
    #                 run_algorithm_SVC_poly_kernel_configuration(metrics, 'SVC with polynomial kernel - degree = ' + str(
    #                     degree) + ', gamma = ' + str(gamma + i) + ', coef0 = ' + str(coef0), X, y, coef0=coef0,
    #                                                             degree=degree, gamma=(gamma + i),
    #                                                             class_weight='balanced', stratify=stratify,
    #                                                             train_size=train_size)
    #         metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_SVM_with_polynomial_kernel)
    #
    # # CALIBRATING tol
    # for tol in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
    #     run_algorithm_SVC_poly_kernel_configuration(metrics, 'SVC with polynomial kernel - tol = ' + str(tol), X, y,
    #                                                 tol=tol, class_weight='balanced', stratify=stratify,
    #                                                 train_size=train_size)
    #     run_algorithm_SVC_poly_kernel_configuration(metrics,
    #                                                 'SVC with polynomial kernel - shrinking = False, tol = ' + str(tol),
    #                                                 X, y, tol=tol, class_weight='balanced', shrinking=False,
    #                                                 stratify=stratify,
    #                                                 train_size=train_size)
    #
    # # CALIBRATING C
    # for C in [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 100]:
    #     run_algorithm_SVC_poly_kernel_configuration(metrics, 'SVC with polynomial kernel - C = ' + str(C), X, y, C=C,
    #                                                 class_weight='balanced',
    #                                                 stratify=stratify, train_size=train_size)
    #     run_algorithm_SVC_poly_kernel_configuration(metrics,
    #                                                 'SVC with polynomial kernel - shrinking = False, C = ' + str(C), X,
    #                                                 y, C=C, shrinking=False, class_weight='balanced', stratify=stratify,
    #                                                 train_size=train_size)
    #
    # # CALIBRATING class_weight
    # run_algorithm_SVC_poly_kernel_configuration(metrics, 'SVC with polynomial kernel - class_weight = balanced', X, y,
    #                                             class_weight='balanced', stratify=stratify, train_size=train_size)
    # run_algorithm_SVC_poly_kernel_configuration(metrics,
    #                                             'SVC with polynomial kernel - shrinking = False, class_weight = balanced',
    #                                             X, y, class_weight='balanced', shrinking=False, stratify=stratify,
    #                                             train_size=train_size)
    #
    # # CALIBRATING max_iter
    # for max_iter in [100, 200, 300, 400, 500, 600, 800, 900, 925, 950, 975, 1000, 1025, 1050, 1075, 1100, 1200, 1300,
    #                  1400, 1500, 1600, 1700, 1800, 1900, 2000]:
    #     run_algorithm_SVC_poly_kernel_configuration(metrics, 'SVC with polynomial kernel - max_iter = ' + str(max_iter),
    #                                                 X, y, max_iter=max_iter, class_weight='balanced', stratify=stratify,
    #                                                 train_size=train_size)
    #     run_algorithm_SVC_poly_kernel_configuration(metrics,
    #                                                 'SVC with polynomial kernel - shrinking = False, max_iter = ' + str(
    #                                                     max_iter), X, y, shrinking=False, max_iter=max_iter,
    #                                                 class_weight='balanced',
    #                                                 stratify=stratify, train_size=train_size)
    #
    # # CALIBRATING cache_size
    # for cache_size in [100, 125, 150, 175, 190, 200, 210, 220, 230, 240, 250, 275, 300, 325, 350, 375, 400, 450, 500,
    #                    550, 600, 650, 700, 750, 800]:
    #     run_algorithm_SVC_poly_kernel_configuration(metrics,
    #                                                 'SVC with polynomial kernel - cache_size = ' + str(cache_size), X,
    #                                                 y, cache_size=cache_size, class_weight='balanced',
    #                                                 stratify=stratify, train_size=train_size)
    #     run_algorithm_SVC_poly_kernel_configuration(metrics,
    #                                                 'SVC with polynomial kernel - shrinking = False, cache_size = ' + str(
    #                                                     cache_size), X, y, shrinking=False, cache_size=cache_size,
    #                                                 class_weight='balanced',
    #                                                 stratify=stratify, train_size=train_size)
    # metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_SVM_with_polynomial_kernel)

    # TODO: our proposed POLYNOMIAL SVC
    for shrinking in [True, False]:
        for degree in range(1, 20, 1):
            for cache_size in range(195, 205, 1):
                for gamma in chain(np.random.uniform(low=0.001, high=1.0, size=(20,))):
                    for coef0 in chain(np.random.uniform(low=2, high=15, size=(20,)),
                                       range(15, 45, 1), range(95, 105), range(145, 155),
                                       range(170, 180), range(195, 205), range(295, 305)):
                        run_algorithm_SVC_poly_kernel_configuration(metrics,
                                                                    'SVC with polynomial kernel - shrinking = ' + str(
                                                                        shrinking) +
                                                                    ', cache_size = ' + str(
                                                                        cache_size) + ', degree=' + str(degree)
                                                                    + ', gamma=' + str(gamma) + ', coef0=' + str(coef0),
                                                                    X, y, shrinking=shrinking, cache_size=cache_size,
                                                                    degree=degree, gamma=gamma, coef0=coef0,
                                                                    class_weight='balanced',
                                                                    stratify=stratify, train_size=train_size)
                metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_SVM_with_polynomial_kernel)
    # # export metrics to CSV FILE
    # df_metrics = pd.DataFrame(metrics)
    # df_metrics.to_csv(filename, encoding='utf-8', index=True)
