import os
import traceback
import warnings
from itertools import chain
from multiprocessing import Manager, Pool

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from data_pre import split_data_in_testing_training, load_normalized_dataset

warnings.filterwarnings("error")

from utils import prediction, cal_metrics, appendMetricsTOCSV, listener_write_to_file, convert_metrics_to_csv


def run_algorithm_gradient_boost_configuration_parallel(X, y, q_metrics,
                                               loss='deviance',
                                               learning_rate=0.1,
                                               n_estimators=100,
                                               subsample=1.0,
                                               criterion='friedman_mse',
                                               min_samples_split=2,
                                               min_samples_leaf=1,
                                               min_weight_fraction_leaf=0.0,
                                               max_depth=3,
                                               min_impurity_decrease=0.0,
                                               init=None,
                                               max_features=None,
                                               max_leaf_nodes=None,
                                               validation_fraction=0.1,
                                               n_iter_no_change=None,
                                               tol=1e-4,
                                               ccp_alpha=0.0,
                                               stratify=False, train_size=0.8):
    X_test, X_train, y_test, y_train = split_data_in_testing_training(X, y, stratify, train_size)
    try:
        # Creating the classifier object
        classifier = GradientBoostingClassifier(loss=loss, learning_rate=learning_rate,
                                                n_estimators=n_estimators, subsample=subsample,
                                                criterion=criterion, min_samples_split=min_samples_split,
                                                min_samples_leaf=min_samples_leaf,
                                                min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                max_depth=max_depth, min_impurity_decrease=min_impurity_decrease,
                                                init=init, max_features=max_features,
                                                max_leaf_nodes=max_leaf_nodes, validation_fraction=validation_fraction,
                                                n_iter_no_change=n_iter_no_change, tol=tol,
                                                ccp_alpha=ccp_alpha,
                                                )
        # Performing training
        classifier.fit(X_train, y_train)

        # Make predictions
        y_pred, y_pred_probabilities = prediction(X_test, classifier)

        # Compute metrics
        label = create_label_for_XGB(loss, learning_rate, n_estimators, subsample, criterion, min_samples_split,
                                     min_samples_leaf, min_weight_fraction_leaf, max_depth, min_impurity_decrease, init,
                                     max_features, max_leaf_nodes, validation_fraction, n_iter_no_change, tol, ccp_alpha)
        precision, recall, f1, roc_auc = cal_metrics(y_test, y_pred, y_pred_probabilities, label, classifier)
        string_results_for_queue = convert_metrics_to_csv(',', label,
                                                          loss, learning_rate, n_estimators, subsample, criterion,
                                                          min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
                                                          max_depth, min_impurity_decrease, init, max_features,
                                                          max_leaf_nodes, validation_fraction, n_iter_no_change, tol,
                                                          ccp_alpha, precision, recall, f1, roc_auc)
        q_metrics.put(string_results_for_queue)
    except Exception as er:
        # pass
        print(er)
        # traceback.print_exc()
        # print(traceback.format_exc())
    except RuntimeWarning as warn:
        # pass
        print(warn)


def create_label_for_XGB(loss, learning_rate, n_estimators, subsample, criterion,
                         min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
                         max_depth, min_impurity_decrease, init, max_features,
                         max_leaf_nodes, validation_fraction, n_iter_no_change, tol,
                         ccp_alpha):
    return "XGB, loss="+loss+", learning_rate=" + str(learning_rate)+", n_estimators=" + str(n_estimators) +", subsample="+\
        str(subsample)+", criterion="+criterion+", min_samples_split="+str(min_samples_split)+", min_samples_leaf="+\
        str(min_samples_leaf)+", min_weight_fraction_leaf="+str(min_weight_fraction_leaf) +", max_depth="+str(max_depth)+\
        ", min_impurity_decrease="+str(min_impurity_decrease)+", init="+str(init)+", max_features="+str(max_features)+\
        ", max_leaf_nodes="+ str(max_leaf_nodes)+", validation_fraction="+str(validation_fraction)+ ", n_iter_no_change="+\
        str(n_iter_no_change)+ ", ccp_alpha="+str(ccp_alpha) + ", tol="+str(tol)


def run_algorithm_gradient_boost_configuration(metrics, label, X, y,
                                               loss='deviance',
                                               learning_rate=0.1,
                                               n_estimators=100,
                                               subsample=1.0,
                                               criterion='friedman_mse',
                                               min_samples_split=2,
                                               min_samples_leaf=1,
                                               min_weight_fraction_leaf=0.0,
                                               max_depth=3,
                                               min_impurity_decrease=0.0,
                                               init=None,
                                               max_features=None,
                                               max_leaf_nodes=None,
                                               validation_fraction=0.1,
                                               n_iter_no_change=None,
                                               tol=1e-4,
                                               ccp_alpha=0.0,
                                               stratify=False, train_size=0.8):
    X_test, X_train, y_test, y_train = split_data_in_testing_training(X, y, stratify, train_size)
    try:
        # Creating the classifier object
        classifier = GradientBoostingClassifier(loss=loss, learning_rate=learning_rate,
                                                n_estimators=n_estimators, subsample=subsample,
                                                criterion=criterion, min_samples_split=min_samples_split,
                                                min_samples_leaf=min_samples_leaf,
                                                min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                max_depth=max_depth, min_impurity_decrease=min_impurity_decrease,
                                                init=init, max_features=max_features,
                                                max_leaf_nodes=max_leaf_nodes, validation_fraction=validation_fraction,
                                                n_iter_no_change=n_iter_no_change, tol=tol,
                                                ccp_alpha=ccp_alpha,
                                                )
        # Performing training
        classifier.fit(X_train, y_train)

        # Make predictions
        y_pred, y_pred_probabilities = prediction(X_test, classifier)

        # Compute metrics
        precision, recall, f1, roc_auc = cal_metrics(y_test, y_pred, y_pred_probabilities, label)
        metrics['label'].append(label)

        metrics['loss'].append(loss)
        metrics['learning_rate'].append(learning_rate)
        metrics['n_estimators'].append(n_estimators)
        metrics['subsample'].append(subsample)
        metrics['criterion'].append(criterion)
        metrics['min_samples_split'].append(min_samples_split)
        metrics['min_samples_leaf'].append(min_samples_leaf)
        metrics['min_weight_fraction_leaf'].append(min_weight_fraction_leaf)
        metrics['max_depth'].append(max_depth)
        metrics['min_impurity_decrease'].append(min_impurity_decrease)
        metrics['init'].append(init)
        metrics['max_features'].append(max_features)
        metrics['max_leaf_nodes'].append(max_leaf_nodes)
        metrics['validation_fraction'].append(validation_fraction)
        metrics['n_iter_no_change'].append(n_iter_no_change)
        metrics['tol'].append(tol)
        metrics['ccp_alpha'].append(ccp_alpha)

        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1_score'].append(f1)
        metrics['roc_auc'].append(roc_auc)
    except Exception as er:
        pass
        # print(er)
    except RuntimeWarning as warn:
        pass
        # print(warn)


def init_metrics_for_XGB():
    return {'label': [],
            'loss': [], 'learning_rate': [], 'n_estimators': [], 'subsample': [],
            'criterion': [], 'min_samples_split': [], 'min_samples_leaf': [], 'min_weight_fraction_leaf': [],
            'max_depth': [], 'min_impurity_decrease': [], 'init': [], 'max_features': [],
            'max_leaf_nodes': [], 'validation_fraction': [], 'n_iter_no_change': [],
            'tol': [], 'ccp_alpha': [],
            'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': []
            }


def run_algorithm_xgb(filename='', path='', stratify=False, train_size=0.8,
                      normalize_data=False, scaler='min-max'):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_XGB()

    # full_path_filename = '/content/drive/MyDrive/code/' + filename
    # path_to_script = os.path.dirname(os.path.abspath(__file__))
    my_filename = os.path.join(path, 'results/xgb', filename)

    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_XGB, header=True)

    # default algorithm
    # run_algorithm_gradient_boost_configuration(metrics, 'Gradient Boosting - default params', X, y,
    #                                            stratify=stratify, train_size=train_size)
    # run_algorithm_gradient_boost_configuration(metrics, 'Gradient Boosting - loss = exponential', X, y,
    #                                            loss='exponential', stratify=stratify,
    #                                            train_size=train_size)
    # run_algorithm_gradient_boost_configuration(metrics,
    #                                            'Gradient Boosting - loss = exponential, criterion = squared_error',
    #                                            X, y, criterion='squared_error', loss='exponential',
    #                                            stratify=stratify, train_size=train_size)
    # run_algorithm_gradient_boost_configuration(metrics, 'Gradient Boosting - criterion = squared_error', X, y,
    #                                            criterion='squared_error', stratify=stratify,
    #                                            train_size=train_size)
    #
    # metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_XGB)
    # # CALIBRATING n_estimators
    # for n_estimators in chain(range(1, 100, 2), range(100, 200, 5), range(200, 300, 10), range(300, 500, 25),
    #                           range(500, 1000, 50)):
    #     run_algorithm_gradient_boost_configuration(metrics, 'Gradient Boosting - n_estimators = ' + str(n_estimators),
    #                                                X, y, n_estimators=n_estimators, stratify=stratify,
    #                                                train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - loss = exponential, n_estimators = ' + str(
    #                                                    n_estimators), X, y, n_estimators=n_estimators,
    #                                                loss='exponential', stratify=stratify, train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - loss = exponential, criterion = squared_error, n_estimators = ' + str(
    #                                                    n_estimators), X, y, n_estimators=n_estimators,
    #                                                criterion='squared_error', loss='exponential', stratify=stratify,
    #                                                train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - criterion = squared_error, n_estimators = ' + str(
    #                                                    n_estimators), X, y, n_estimators=n_estimators,
    #                                                criterion='squared_error', stratify=stratify, train_size=train_size)
    #
    #     # CALIBRATING learning_rate
    # for learning_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3,
    #                       1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
    #                       2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.5, 6.0, 6.5, 7.0,
    #                       7.5, 8.0, 8.5, 9.0, 9.5, 10.0]:
    #     run_algorithm_gradient_boost_configuration(metrics, 'Gradient Boosting - learning_rate = ' + str(learning_rate),
    #                                                X, y, learning_rate=(0.0 + learning_rate), stratify=stratify,
    #                                                train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - loss = exponential, learning_rate = ' + str(
    #                                                    learning_rate), X, y, learning_rate=(0.0 + learning_rate),
    #                                                loss='exponential', stratify=stratify, train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - loss = exponential, criterion = squared_error, learning_rate = ' + str(
    #                                                    learning_rate), X, y, learning_rate=(0.0 + learning_rate),
    #                                                criterion='squared_error', loss='exponential', stratify=stratify,
    #                                                train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - criterion = squared_error, learning_rate = ' + str(
    #                                                    learning_rate), X, y, learning_rate=(0.0 + learning_rate),
    #                                                criterion='squared_error', stratify=stratify, train_size=train_size)
    #
    # for learning_rate in chain(range(10, 50, 1), range(50, 100, 2), range(100, 250, 5), range(250, 500, 10),
    #                            range(500, 1000, 25)):
    #     run_algorithm_gradient_boost_configuration(metrics, 'Gradient Boosting - learning_rate = ' + str(learning_rate),
    #                                                X, y, learning_rate=(0.0 + learning_rate), stratify=stratify,
    #                                                train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - loss = exponential, learning_rate = ' + str(
    #                                                    learning_rate), X, y, learning_rate=(0.0 + learning_rate),
    #                                                loss='exponential', stratify=stratify, train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - loss = exponential, criterion = squared_error, learning_rate = ' + str(
    #                                                    learning_rate), X, y, learning_rate=(0.0 + learning_rate),
    #                                                criterion='squared_error', loss='exponential', stratify=stratify,
    #                                                train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - criterion = squared_error, learning_rate = ' + str(
    #                                                    learning_rate), X, y, learning_rate=(0.0 + learning_rate),
    #                                                criterion='squared_error', stratify=stratify, train_size=train_size)
    #
    # # CALIBRATING subsample
    # for subsample in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.875, 0.9, 0.905, 0.91, 0.915, 0.92,
    #                   0.925, 0.93, 0.935, 0.94, 0.945, 0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995]:
    #     run_algorithm_gradient_boost_configuration(metrics, 'Gradient Boosting - subsample = ' + str(subsample), X, y,
    #                                                subsample=subsample, stratify=stratify, train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - loss = exponential, subsample = ' + str(
    #                                                    subsample), X, y, subsample=subsample, loss='exponential',
    #                                                stratify=stratify, train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - loss = exponential, criterion = squared_error, subsample = ' + str(
    #                                                    subsample), X, y, subsample=subsample, criterion='squared_error',
    #                                                loss='exponential', stratify=stratify, train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - criterion = squared_error, subsample = ' + str(
    #                                                    subsample), X, y, subsample=subsample, criterion='squared_error',
    #                                                stratify=stratify, train_size=train_size)
    #
    #     # CALIBRATING min_samples_split & min_samples_leaf & ccp_alpha & min_impurity_decrease
    # for min_samples_split in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.875, 0.9, 0.905, 0.91, 0.915,
    #                           0.92, 0.925, 0.93, 0.935, 0.94, 0.945, 0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985,
    #                           0.99, 0.995, 1.0]:
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - min_samples_split = ' + str(min_samples_split),
    #                                                X, y, min_samples_split=min_samples_split, stratify=stratify,
    #                                                train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - loss = exponential, min_samples_split = ' + str(
    #                                                    min_samples_split), X, y, min_samples_split=min_samples_split,
    #                                                loss='exponential', stratify=stratify, train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - loss = exponential, criterion = squared_error, min_samples_split = ' + str(
    #                                                    min_samples_split), X, y, min_samples_split=min_samples_split,
    #                                                criterion='squared_error', loss='exponential', stratify=stratify,
    #                                                train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - criterion = squared_error, min_samples_split = ' + str(
    #                                                    min_samples_split), X, y, min_samples_split=min_samples_split,
    #                                                criterion='squared_error', stratify=stratify, train_size=train_size)
    #
    #     run_algorithm_gradient_boost_configuration(metrics, 'Gradient Boosting - ccp_alpha = ' + str(min_samples_split),
    #                                                X, y, ccp_alpha=min_samples_split, stratify=stratify,
    #                                                train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - loss = exponential, ccp_alpha = ' + str(
    #                                                    min_samples_split), X, y, ccp_alpha=min_samples_split,
    #                                                loss='exponential', stratify=stratify, train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - loss = exponential, criterion = squared_error, ccp_alpha = ' + str(
    #                                                    min_samples_split), X, y, ccp_alpha=min_samples_split,
    #                                                criterion='squared_error', loss='exponential', stratify=stratify,
    #                                                train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - criterion = squared_error, ccp_alpha = ' + str(
    #                                                    min_samples_split), X, y, ccp_alpha=min_samples_split,
    #                                                criterion='squared_error', stratify=stratify, train_size=train_size)
    #
    #     run_algorithm_gradient_boost_configuration(metrics, 'Gradient Boosting - min_impurity_decrease = ' + str(
    #         min_samples_split), X, y, min_impurity_decrease=min_samples_split, stratify=stratify, train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - loss = exponential, min_impurity_decrease = ' + str(
    #                                                    min_samples_split), X, y,
    #                                                min_impurity_decrease=min_samples_split, loss='exponential',
    #                                                stratify=stratify, train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - loss = exponential, criterion = squared_error, min_impurity_decrease = ' + str(
    #                                                    min_samples_split), X, y,
    #                                                min_impurity_decrease=min_samples_split, criterion='squared_error',
    #                                                loss='exponential', stratify=stratify, train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - criterion = squared_error, min_impurity_decrease = ' + str(
    #                                                    min_samples_split), X, y,
    #                                                min_impurity_decrease=min_samples_split, criterion='squared_error',
    #                                                stratify=stratify, train_size=train_size)
    #
    # for min_samples_split in chain(range(3, 20, 1), range(20, 40, 2), range(40, 100, 5), range(100, 200, 10),
    #                                range(200, 500, 25), range(500, 1000, 50)):
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - min_samples_split = ' + str(min_samples_split),
    #                                                X, y, min_samples_split=min_samples_split, stratify=stratify,
    #                                                train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - loss = exponential, min_samples_split = ' + str(
    #                                                    min_samples_split), X, y, min_samples_split=min_samples_split,
    #                                                loss='exponential', stratify=stratify, train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - loss = exponential, criterion = squared_error, min_samples_split = ' + str(
    #                                                    min_samples_split), X, y, min_samples_split=min_samples_split,
    #                                                criterion='squared_error', loss='exponential', stratify=stratify,
    #                                                train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - criterion = squared_error, min_samples_split = ' + str(
    #                                                    min_samples_split), X, y, min_samples_split=min_samples_split,
    #                                                criterion='squared_error', stratify=stratify, train_size=train_size)
    #
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - min_samples_leaf = ' + str(min_samples_split),
    #                                                X, y, min_samples_leaf=min_samples_split, stratify=stratify,
    #                                                train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - loss = exponential, min_samples_leaf = ' + str(
    #                                                    min_samples_split), X, y, min_samples_leaf=min_samples_split,
    #                                                loss='exponential', stratify=stratify, train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - loss = exponential, criterion = squared_error, min_samples_leaf = ' + str(
    #                                                    min_samples_split), X, y, min_samples_leaf=min_samples_split,
    #                                                criterion='squared_error', loss='exponential', stratify=stratify,
    #                                                train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - criterion = squared_error, min_samples_leaf = ' + str(
    #                                                    min_samples_split), X, y, min_samples_leaf=min_samples_split,
    #                                                criterion='squared_error', stratify=stratify, train_size=train_size)
    #
    #     run_algorithm_gradient_boost_configuration(metrics, 'Gradient Boosting - ccp_alpha = ' + str(min_samples_split),
    #                                                X, y, ccp_alpha=min_samples_split, stratify=stratify,
    #                                                train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - loss = exponential, ccp_alpha = ' + str(
    #                                                    min_samples_split), X, y, ccp_alpha=min_samples_split,
    #                                                loss='exponential', stratify=stratify, train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - loss = exponential, criterion = squared_error, ccp_alpha = ' + str(
    #                                                    min_samples_split), X, y, ccp_alpha=min_samples_split,
    #                                                criterion='squared_error', loss='exponential', stratify=stratify,
    #                                                train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - criterion = squared_error, ccp_alpha = ' + str(
    #                                                    min_samples_split), X, y, ccp_alpha=min_samples_split,
    #                                                criterion='squared_error', stratify=stratify, train_size=train_size)
    #
    #     run_algorithm_gradient_boost_configuration(metrics, 'Gradient Boosting - min_impurity_decrease = ' + str(
    #         min_samples_split), X, y, min_impurity_decrease=min_samples_split, stratify=stratify, train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - loss = exponential, min_impurity_decrease = ' + str(
    #                                                    min_samples_split), X, y,
    #                                                min_impurity_decrease=min_samples_split, loss='exponential',
    #                                                stratify=stratify, train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - loss = exponential, criterion = squared_error, min_impurity_decrease = ' + str(
    #                                                    min_samples_split), X, y,
    #                                                min_impurity_decrease=min_samples_split, criterion='squared_error',
    #                                                loss='exponential', stratify=stratify, train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - criterion = squared_error, min_impurity_decrease = ' + str(
    #                                                    min_samples_split), X, y,
    #                                                min_impurity_decrease=min_samples_split, criterion='squared_error',
    #                                                stratify=stratify, train_size=train_size)
    #
    #     # CALIBRATING min_weight_fraction_leaf (0.0, 0.5]
    # for min_weight_fraction_leaf in [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07,
    #                                  0.075, 0.08, 0.085, 0.09, 0.095, 0.1,
    #                                  0.11, 0.115, 0.12, 0.125, 0.13, 0.135, 0.14, 0.145, 0.15, 0.155, 0.16, 0.165, 0.17,
    #                                  0.175, 0.18, 0.185, 0.19, 0.195, 0.2,
    #                                  0.21, 0.215, 0.22, 0.225, 0.23, 0.235, 0.24, 0.245, 0.25, 0.255, 0.26, 0.265, 0.27,
    #                                  0.275, 0.28, 0.285, 0.29, 0.295, 0.3,
    #                                  0.31, 0.315, 0.32, 0.325, 0.33, 0.335, 0.34, 0.345, 0.35, 0.355, 0.36, 0.365, 0.37,
    #                                  0.375, 0.38, 0.385, 0.39, 0.395, 0.4,
    #                                  0.41, 0.415, 0.42, 0.425, 0.43, 0.435, 0.44, 0.445, 0.45, 0.455, 0.46, 0.465, 0.47,
    #                                  0.475, 0.48, 0.485, 0.49, 0.495, 0.5
    #                                  ]:
    #     run_algorithm_gradient_boost_configuration(metrics, 'Gradient Boosting - min_weight_fraction_leaf = ' + str(
    #         min_weight_fraction_leaf), X, y, min_weight_fraction_leaf=min_weight_fraction_leaf, stratify=stratify,
    #                                                train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - loss = exponential, min_weight_fraction_leaf = ' + str(
    #                                                    min_weight_fraction_leaf), X, y,
    #                                                min_weight_fraction_leaf=min_weight_fraction_leaf,
    #                                                loss='exponential', stratify=stratify, train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - loss = exponential, criterion = squared_error, min_weight_fraction_leaf = ' + str(
    #                                                    min_weight_fraction_leaf), X, y,
    #                                                min_weight_fraction_leaf=min_weight_fraction_leaf,
    #                                                criterion='squared_error', loss='exponential', stratify=stratify,
    #                                                train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - criterion = squared_error, min_weight_fraction_leaf = ' + str(
    #                                                    min_weight_fraction_leaf), X, y,
    #                                                min_weight_fraction_leaf=min_weight_fraction_leaf,
    #                                                criterion='squared_error', stratify=stratify, train_size=train_size)
    #
    #     run_algorithm_gradient_boost_configuration(metrics, 'Gradient Boosting - min_samples_leaf = ' + str(
    #         min_weight_fraction_leaf), X, y, min_samples_leaf=min_weight_fraction_leaf, stratify=stratify,
    #                                                train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - loss = exponential, min_samples_leaf = ' + str(
    #                                                    min_weight_fraction_leaf), X, y,
    #                                                min_samples_leaf=min_weight_fraction_leaf, loss='exponential',
    #                                                stratify=stratify, train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - loss = exponential, criterion = squared_error, min_samples_leaf = ' + str(
    #                                                    min_weight_fraction_leaf), X, y,
    #                                                min_samples_leaf=min_weight_fraction_leaf, criterion='squared_error',
    #                                                loss='exponential', stratify=stratify, train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - criterion = squared_error, min_samples_leaf = ' + str(
    #                                                    min_weight_fraction_leaf), X, y,
    #                                                min_samples_leaf=min_weight_fraction_leaf, criterion='squared_error',
    #                                                stratify=stratify, train_size=train_size)
    #
    #     # CALIBRATING max_depth [1, inf)
    # for max_depth in chain(range(1, 20, 1), range(20, 50, 2), range(50, 100, 5), range(100, 250, 10),
    #                        range(250, 500, 25), range(500, 1000, 50)):
    #     run_algorithm_gradient_boost_configuration(metrics, 'Gradient Boosting - max_depth = ' + str(max_depth), X, y,
    #                                                max_depth=max_depth, stratify=stratify, train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - loss = exponential, max_depth = ' + str(
    #                                                    max_depth), X, y, max_depth=max_depth, loss='exponential',
    #                                                stratify=stratify, train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - loss = exponential, criterion = squared_error, max_depth = ' + str(
    #                                                    max_depth), X, y, max_depth=max_depth, criterion='squared_error',
    #                                                loss='exponential', stratify=stratify, train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - criterion = squared_error, max_depth = ' + str(
    #                                                    max_depth), X, y, max_depth=max_depth, criterion='squared_error',
    #                                                stratify=stratify, train_size=train_size)
    #
    #     # CALIBRATING init 'zero' or DecisionTree etc. other estimators TODO
    # run_algorithm_gradient_boost_configuration(metrics, 'Gradient Boosting - init = zero', X, y, init='zero',
    #                                            stratify=stratify, train_size=train_size)
    # run_algorithm_gradient_boost_configuration(metrics, 'Gradient Boosting - loss = exponential, init = zero', X, y,
    #                                            init='zero', loss='exponential', stratify=stratify,
    #                                            train_size=train_size)
    # run_algorithm_gradient_boost_configuration(metrics,
    #                                            'Gradient Boosting - loss = exponential, criterion = squared_error, init = zero',
    #                                            X, y, init='zero', criterion='squared_error', loss='exponential',
    #                                            stratify=stratify, train_size=train_size)
    # run_algorithm_gradient_boost_configuration(metrics, 'Gradient Boosting - criterion = squared_error, init = zero', X,
    #                                            y, init='zero', criterion='squared_error', stratify=stratify,
    #                                            train_size=train_size)
    # # TODO more estimators
    # metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_XGB)
    # # CALIBRATING max_features (0.0, 1.0], [1, n_features) {‘auto’, ‘sqrt’, ‘log2’}
    # run_algorithm_gradient_boost_configuration(metrics, 'Gradient Boosting - max_features = auto', X, y,
    #                                            max_features='auto', stratify=stratify, train_size=train_size)
    # run_algorithm_gradient_boost_configuration(metrics, 'Gradient Boosting - loss = exponential, max_features = auto',
    #                                            X, y, max_features='auto', loss='exponential', stratify=stratify,
    #                                            train_size=train_size)
    # run_algorithm_gradient_boost_configuration(metrics,
    #                                            'Gradient Boosting - loss = exponential, criterion = squared_error, max_features = auto',
    #                                            X, y, max_features='auto', criterion='squared_error', loss='exponential',
    #                                            stratify=stratify, train_size=train_size)
    # run_algorithm_gradient_boost_configuration(metrics,
    #                                            'Gradient Boosting - criterion = squared_error, max_features = auto', X,
    #                                            y, max_features='auto', criterion='squared_error', stratify=stratify,
    #                                            train_size=train_size)
    #
    # run_algorithm_gradient_boost_configuration(metrics, 'Gradient Boosting - max_features = ‘sqrt’', X, y,
    #                                            max_features='sqrt', stratify=stratify, train_size=train_size)
    # run_algorithm_gradient_boost_configuration(metrics, 'Gradient Boosting - loss = exponential, max_features = ‘sqrt’',
    #                                            X, y, max_features='sqrt', loss='exponential', stratify=stratify,
    #                                            train_size=train_size)
    # run_algorithm_gradient_boost_configuration(metrics,
    #                                            'Gradient Boosting - loss = exponential, criterion = squared_error, max_features = ‘sqrt’',
    #                                            X, y, max_features='sqrt', criterion='squared_error', loss='exponential',
    #                                            stratify=stratify, train_size=train_size)
    # run_algorithm_gradient_boost_configuration(metrics,
    #                                            'Gradient Boosting - criterion = squared_error, max_features = ‘sqrt’',
    #                                            X, y, max_features='sqrt', criterion='squared_error', stratify=stratify,
    #                                            train_size=train_size)
    #
    # run_algorithm_gradient_boost_configuration(metrics, 'Gradient Boosting - max_features = log2', X, y,
    #                                            max_features='log2', stratify=stratify, train_size=train_size)
    # run_algorithm_gradient_boost_configuration(metrics, 'Gradient Boosting - loss = exponential, max_features = log2',
    #                                            X, y, max_features='log2', loss='exponential', stratify=stratify,
    #                                            train_size=train_size)
    # run_algorithm_gradient_boost_configuration(metrics,
    #                                            'Gradient Boosting - loss = exponential, criterion = squared_error, max_features = log2',
    #                                            X, y, max_features='log2', criterion='squared_error', loss='exponential',
    #                                            stratify=stratify, train_size=train_size)
    # run_algorithm_gradient_boost_configuration(metrics,
    #                                            'Gradient Boosting - criterion = squared_error, max_features = log2', X,
    #                                            y, max_features='log2', criterion='squared_error', stratify=stratify,
    #                                            train_size=train_size)
    #
    # for max_features in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.875, 0.9, 0.905, 0.91, 0.915, 0.92,
    #                      0.925, 0.93, 0.935, 0.94, 0.945, 0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99,
    #                      0.995, 1.0]:
    #     run_algorithm_gradient_boost_configuration(metrics, 'Gradient Boosting - max_features = ' + str(max_features),
    #                                                X, y, max_features=max_features, stratify=stratify,
    #                                                train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - loss = exponential, max_features = ' + str(
    #                                                    max_features), X, y, max_features=max_features,
    #                                                loss='exponential', stratify=stratify, train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - loss = exponential, criterion = squared_error, max_features = ' + str(
    #                                                    max_features), X, y, max_features=max_features,
    #                                                criterion='squared_error', loss='exponential', stratify=stratify,
    #                                                train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - criterion = squared_error, max_features = ' + str(
    #                                                    max_features), X, y, max_features=max_features,
    #                                                criterion='squared_error', stratify=stratify, train_size=train_size)
    # for max_feature in range(1, 19, 1):
    #     run_algorithm_gradient_boost_configuration(metrics, 'Gradient Boosting - max_features = ' + str(max_features),
    #                                                X, y, max_features=max_features, stratify=stratify,
    #                                                train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - loss = exponential, max_features = ' + str(
    #                                                    max_features), X, y, max_features=max_features,
    #                                                loss='exponential', stratify=stratify, train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - loss = exponential, criterion = squared_error, max_features = ' + str(
    #                                                    max_features), X, y, max_features=max_features,
    #                                                criterion='squared_error', loss='exponential', stratify=stratify,
    #                                                train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - criterion = squared_error, max_features = ' + str(
    #                                                    max_features), X, y, max_features=max_features,
    #                                                criterion='squared_error', stratify=stratify, train_size=train_size)
    # metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_XGB)
    # # CALIBRATING max_leaf_nodes [2, inf)]
    # for max_leaf_nodes in chain(range(2, 20, 1), range(20, 50, 2), range(50, 100, 5), range(100, 250, 10),
    #                             range(250, 500, 25), range(500, 1000, 50)):
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - max_leaf_nodes = ' + str(max_leaf_nodes), X, y,
    #                                                max_leaf_nodes=max_leaf_nodes, stratify=stratify,
    #                                                train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - loss = exponential, max_leaf_nodes = ' + str(
    #                                                    max_leaf_nodes), X, y, max_leaf_nodes=max_leaf_nodes,
    #                                                loss='exponential', stratify=stratify, train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - loss = exponential, criterion = squared_error, max_leaf_nodes = ' + str(
    #                                                    max_leaf_nodes), X, y, max_leaf_nodes=max_leaf_nodes,
    #                                                criterion='squared_error', loss='exponential', stratify=stratify,
    #                                                train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - criterion = squared_error, max_leaf_nodes = ' + str(
    #                                                    max_leaf_nodes), X, y, max_leaf_nodes=max_leaf_nodes,
    #                                                criterion='squared_error', stratify=stratify, train_size=train_size)
    #
    # # CALIBRATING validation_fraction (0.0, 1.0) and n_iter_no_change [1, inf)
    # for n_iter_no_change in chain(range(1, 20, 1), range(20, 50, 2), range(50, 100, 5), range(100, 250, 10),
    #                               range(250, 500, 25), range(500, 1000, 50)):
    #     for validation_fraction in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.875, 0.9, 0.905, 0.91,
    #                                 0.915, 0.92, 0.925, 0.93, 0.935, 0.94, 0.945, 0.95, 0.955, 0.96, 0.965, 0.97, 0.975,
    #                                 0.98, 0.985, 0.99, 0.995, 1.0]:
    #         try:
    #             run_algorithm_gradient_boost_configuration(metrics, 'Gradient Boosting - n_iter_no_change = ' + str(
    #                 n_iter_no_change) + ', validation_fraction = ' + str(validation_fraction), X, y,
    #                                                        n_iter_no_change=n_iter_no_change,
    #                                                        validation_fraction=validation_fraction, stratify=stratify,
    #                                                        train_size=train_size)
    #             run_algorithm_gradient_boost_configuration(metrics,
    #                                                        'Gradient Boosting - loss = exponential, n_iter_no_change = ' + str(
    #                                                            n_iter_no_change) + ', validation_fraction = ' + str(
    #                                                            validation_fraction), X, y,
    #                                                        n_iter_no_change=n_iter_no_change,
    #                                                        validation_fraction=validation_fraction, loss='exponential',
    #                                                        stratify=stratify, train_size=train_size)
    #             run_algorithm_gradient_boost_configuration(metrics,
    #                                                        'Gradient Boosting - loss = exponential, criterion = squared_error, n_iter_no_change = ' + str(
    #                                                            n_iter_no_change) + ', validation_fraction = ' + str(
    #                                                            validation_fraction), X, y,
    #                                                        n_iter_no_change=n_iter_no_change,
    #                                                        validation_fraction=validation_fraction,
    #                                                        criterion='squared_error', loss='exponential',
    #                                                        stratify=stratify, train_size=train_size)
    #             run_algorithm_gradient_boost_configuration(metrics,
    #                                                        'Gradient Boosting - criterion = squared_error, n_iter_no_change = ' + str(
    #                                                            n_iter_no_change) + ', validation_fraction = ' + str(
    #                                                            validation_fraction), X, y,
    #                                                        n_iter_no_change=n_iter_no_change,
    #                                                        validation_fraction=validation_fraction,
    #                                                        criterion='squared_error', stratify=stratify,
    #                                                        train_size=train_size)
    #         except:
    #             pass
    #
    # # CALIBRATING tol
    # for tol in [1e-1, 1e-2, 1e-3, 1e-5, 1e-6, 1e-7]:
    #     run_algorithm_gradient_boost_configuration(metrics, 'Gradient Boosting - tol = ' + str(tol), X, y, tol=tol,
    #                                                stratify=stratify, train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics, 'Gradient Boosting - loss = exponential, tol = ' + str(tol),
    #                                                X, y, tol=tol, loss='exponential', stratify=stratify,
    #                                                train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - loss = exponential, criterion = squared_error, tol = ' + str(
    #                                                    tol), X, y, tol=tol, criterion='squared_error',
    #                                                loss='exponential', stratify=stratify, train_size=train_size)
    #     run_algorithm_gradient_boost_configuration(metrics,
    #                                                'Gradient Boosting - criterion = squared_error, tol = ' + str(tol),
    #                                                X, y, tol=tol, criterion='squared_error', stratify=stratify,
    #                                                train_size=train_size)
    #
    # metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_XGB)

    # min samples split	145-155	2	95-105	45-55
    # n iter no change	8..16	None	27-37
    # validation fraction	0.1-0.8
    # min samples leaf	90-100	5..10	0.1-0.2	1
    # max leaf nodes	19-33	397-403	40-49	None

    for criterion in ['squared_error', 'friedman_mse']:
        for loss in ['exponential', 'log_loss']:
            for learning_rate in chain(np.random.uniform(low=0.05, high=0.15, size=(5,)),
                                       np.random.uniform(low=0.7, high=0.9, size=(10,))):
                for n_estimators in chain(range(71, 111, 1), range(120, 140, 1)):
                    for subsample in [0.4, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39,
                                      0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.9,
                                      0.91, 0.92, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]:
                        for min_samples_split in chain(range(1, 21, 1), [0.6, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55,
                                                                         0.56, 0.57, 0.58, 0.59, 0.61, 0.62, 0.63,
                                                                         0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7]):
                            for min_samples_leaf in chain(range(1, 16, 1), np.random.uniform(0.1, 1.5, (10,))):
                                for max_leaf_nodes in chain([None], range(2, 16, 1), range(50, 60, 1),
                                                            range(175, 185, 1),
                                                            range(245, 255, 1)):
                                    for max_depth in range(1, 10, 1):
                                        for min_weight_fraction_leaf in [0.0, 0.011, 0.012, 0.013, 0.014, 0.015,
                                                                         0.016, 0.017, 0.018, 0.019, 0.03, 0.031, 0.032,
                                                                         0.033, 0.034, 0.035, 0.036, 0.037, 0.038,
                                                                         0.039,
                                                                         0.04, 0.041, 0.042, 0.043, 0.045, 0.046, 0.047,
                                                                         0.048, 0.049, 0.05]:
                                            for max_features in [None, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87,
                                                                 0.88, 0.89, 0.9]:
                                                for validation_fraction in [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11,
                                                                            0.12,
                                                                            0.13, 0.14, 0.15, 0.25, 0.26, 0.27, 0.28,
                                                                            0.29, 0.3, 0.31,
                                                                            0.32, 0.33, 0.34, 0.35]:
                                                    for n_iter_no_change in chain([None], range(5, 16, 1),
                                                                                  range(445, 455, 1),
                                                                                  range(645, 655, 1), range(845, 855)):
                                                        run_algorithm_gradient_boost_configuration(metrics,
                                                                                                   'Gradient Boosting - criterion = ' + criterion
                                                                                                   + ', loss = ' + str(
                                                                                                       loss) + ', learning_rate=' +
                                                                                                   str(learning_rate) + ', n_estimators=' +
                                                                                                   str(n_estimators) + ', subsample=' + str(
                                                                                                       subsample) +
                                                                                                   ', min_samples_split=' +
                                                                                                   str(min_samples_split) + ', min_samples_leaf' +
                                                                                                   str(min_samples_leaf) + ', max_leaf_nodes=' +
                                                                                                   str(max_leaf_nodes) + ', max_depth=' + str(
                                                                                                       max_depth) +
                                                                                                   ', min_weight_fraction_leaf=' + str(
                                                                                                       min_weight_fraction_leaf) +
                                                                                                   ', max_features = ' + str(
                                                                                                       max_features) +
                                                                                                   ', validation_fraction=' + str(
                                                                                                       validation_fraction) +
                                                                                                   ', n_iter_no_change=' + str(
                                                                                                       n_iter_no_change),
                                                                                                   X, y,
                                                                                                   criterion=criterion,
                                                                                                   loss=loss,
                                                                                                   learning_rate=learning_rate,
                                                                                                   n_estimators=n_estimators,
                                                                                                   subsample=subsample,
                                                                                                   min_samples_split=min_samples_split,
                                                                                                   min_samples_leaf=min_samples_leaf,
                                                                                                   max_leaf_nodes=max_leaf_nodes,
                                                                                                   max_depth=max_depth,
                                                                                                   min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                                                                   max_features=max_features,
                                                                                                   validation_fraction=validation_fraction,
                                                                                                   n_iter_no_change=n_iter_no_change,
                                                                                                   stratify=stratify,
                                                                                                   train_size=train_size)
                                                    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_XGB)

    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_XGB)


def run_algorithm_xgb_parallel(filename='', path='', stratify=False, train_size=0.8,
                               normalize_data=False, scaler='min-max', no_threads=8):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_XGB()
    my_filename = os.path.join(path, 'results/xgb', filename)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_XGB, header=True)

    # criterion_list = [ 'squared_error', 'friedman_mse']
    criterion_list = ['friedman_mse']
    loss_list= ['exponential', 'log_loss']
    learning_rate_list = list(chain(np.random.uniform(low=0.05, high=0.15, size=(3,)),
                               np.random.uniform(low=0.7, high=0.9, size=(3,))))
    n_estimators_list = [70, 80, 90, 100, 120, 130, 140]
    min_samples_split_list = list(chain(range(2, 21, 5), [0.6, 0.5, 0.7]))
    min_samples_leaf_list = list(chain([1, 2, 5, 10, 15], np.random.uniform(0.1, 0.5, (3,))))
    max_leaf_nodes_list = list(chain([None], [2, 5, 10, 15]))
    max_depth_list = [1, 3, 5, 7, 9]
    # 117600

    with Manager() as manager:
        q_metrics = manager.Queue()
        jobs = []

        with Pool(no_threads) as pool:
            watcher = pool.apply_async(listener_write_to_file, (q_metrics, my_filename))
            for criterion in criterion_list:
                for loss in loss_list:
                    for learning_rate in learning_rate_list:
                        for n_estimators in n_estimators_list:
                            for min_samples_split in min_samples_split_list:
                                for min_samples_leaf in min_samples_leaf_list:
                                    for max_leaf_nodes in max_leaf_nodes_list:
                                        for max_depth in max_depth_list:
                                            job = pool.apply_async(run_algorithm_gradient_boost_configuration_parallel,
                                                                   (X, y, q_metrics, loss, learning_rate, n_estimators, 1.0,
                                                                    criterion, min_samples_split, min_samples_leaf, 0.0, max_depth,
                                                                    0.0, None, None, max_leaf_nodes, 0.1, None, 1e-4, 0.0,
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
