import os
import warnings
from itertools import chain
from multiprocessing import Manager, Pool

import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope, EmpiricalCovariance, GraphicalLassoCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from data_post import compute_average_metric
from data_pre import split_data_in_testing_training, load_normalized_dataset

warnings.filterwarnings("error")
from utils import prediction, cal_metrics, appendMetricsTOCSV, convert_metrics_to_csv, listener_write_to_file


def create_LDA_classifier(row):
    # TODO refactor and reuse preprocess params
    if (row['shrinkage'] == 'None' or row['shrinkage'] == None or str(row['shrinkage']) == 'nan'):
        shrinkage = None
    elif (row['shrinkage'] == 'auto'):
        shrinkage = 'auto'
    else:
        shrinkage = float(row['shrinkage'])

    if (row['n_components'] == None or row['n_components'] == 'None' or str(row['n_components']) == 'nan'):
        n_components = None
    else:
        n_components = int(row['n_components'])

    if (row['covariance_estimator'] == "EmpiricalCovariance(store_precision=True, assume_centered=False)"):
        covariance_estimator = EmpiricalCovariance(store_precision=True, assume_centered=False)
    elif (row['covariance_estimator'] == "EmpiricalCovariance(store_precision=False, assume_centered=False)"):
        covariance_estimator = EmpiricalCovariance(store_precision=False, assume_centered=False)
    else:
        covariance_estimator = None
    classifier = LinearDiscriminantAnalysis(tol=float(row['tol']),
                                            solver=row['solver'], shrinkage=shrinkage,
                                            store_covariance=bool(row['store_covariance']),
                                            n_components=n_components, covariance_estimator=covariance_estimator)
    return classifier

def run_algorithm_qda_configuration(metrics, label, X, y, tol=1e-4,
                                    stratify=False, train_size=0.8):
    X_train, X_test, y_train, y_test = split_data_in_testing_training(X, y, stratify, train_size)

    try:
        # Creating the classifier object
        classifier = QuadraticDiscriminantAnalysis(tol=tol)

        # Performing training
        classifier.fit(X_train, y_train)

        # Make predictions
        y_pred, y_pred_probabilities = prediction(X_test, classifier)

        # Compute metrics
        precision, recall, f1, roc_auc = cal_metrics(y_test, y_pred, y_pred_probabilities, label, classifier)
        metrics['label'].append(label)

        metrics['tol'].append(tol)

        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1_score'].append(f1)
        metrics['roc_auc'].append(roc_auc)
    except Exception as err:
        print(err)
    except RuntimeWarning as warn:
        print(warn)


def init_metrics_for_QDA():
    return {'label': [], 'tol': [],
            'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': []
            }


def run_algorithm_qda(filename='', path='', stratify=False, train_size=0.8,
                      normalize_data=False, scaler='min-max'):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_QDA()

    # full_path_filename = '/content/drive/MyDrive/code/' + filename
    # path_to_script = os.path.dirname(os.path.abspath(__file__))
    my_filename = os.path.join(path, 'results/qlda', filename)

    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_QDA, header=True)

    # default algorithm
    run_algorithm_qda_configuration(metrics, 'QDA - default params', X, y, stratify=stratify, train_size=train_size,
                                    )

    # CALIBRATING tol
    for tol in [1e-2, 1e-3, 1e-5, 1e-6, 1e-7]:
        run_algorithm_qda_configuration(metrics, 'QDA: tol = ' + str(tol), X, y, tol=tol,
                                        stratify=stratify, train_size=train_size)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_QDA)
    for tol in np.random.uniform(low=1e-7, high=0.02, size=(1000,)):
        run_algorithm_qda_configuration(metrics, 'QDA: tol = ' + str(tol), X, y, tol=tol,
                                        stratify=stratify, train_size=train_size)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_QDA)


def run_algorithm_lda_configuration_parallel(X, y, q_metrics,
                                             tol=1e-4, solver='svd', shrinkage=None, store_covariance=False,
                                             n_components=None, covariance_estimator=None,
                                             stratify=False, train_size=0.8
                                             ):
    X_train, X_test, y_train, y_test = split_data_in_testing_training(X, y, stratify, train_size)
    try:
        # Creating the classifier object
        classifier = LinearDiscriminantAnalysis(tol=tol, solver=solver, shrinkage=shrinkage,
                                                n_components=n_components, store_covariance=store_covariance,
                                                covariance_estimator=covariance_estimator)
        # Performing training
        classifier.fit(X_train, y_train)

        # Make predictions
        y_pred, y_pred_probabilities = prediction(X_test, classifier)

        # Compute metrics
        label = create_label_LDA(tol, solver, shrinkage, store_covariance, n_components, covariance_estimator)
        precision, recall, f1, roc_auc = cal_metrics(y_test, y_pred, y_pred_probabilities, label, classifier)
        string_results_for_queue = convert_metrics_to_csv(',', label, tol, solver, shrinkage, n_components,
                                                          store_covariance, covariance_estimator,
                                                          precision, recall, f1, roc_auc)
        q_metrics.put(string_results_for_queue)
    except Exception as err:
        print("ERROr: " + str(err))
    except RuntimeWarning as warn:
        pass


def run_algorithm_lda_configuration(metrics, label, X, y, tol=1e-4,
                                    solver='svd', shrinkage=None, store_covariance=False,
                                    n_components=None, covariance_estimator=None,
                                    stratify=False, train_size=0.8
                                    ):
    X_train, X_test, y_train, y_test = split_data_in_testing_training(X, y, stratify, train_size)
    try:
        # Creating the classifier object
        classifier = LinearDiscriminantAnalysis(tol=tol, solver=solver, shrinkage=shrinkage,
                                                n_components=n_components, store_covariance=store_covariance,
                                                covariance_estimator=covariance_estimator)

        # Performing training
        classifier.fit(X_train, y_train)

        # Make predictions
        y_pred, y_pred_probabilities = prediction(X_test, classifier)

        # Compute metrics
        precision, recall, f1, roc_auc = cal_metrics(y_test, y_pred, y_pred_probabilities, label, classifier)
        metrics['label'].append(label)

        metrics['tol'].append(tol)
        metrics['solver'].append(solver)
        metrics['shrinkage'].append(shrinkage)
        metrics['n_components'].append(n_components)
        metrics['store_covariance'].append(store_covariance)
        metrics['covariance_estimator'].append(covariance_estimator)

        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1_score'].append(f1)
        metrics['roc_auc'].append(roc_auc)
    except Exception as err:
        print("ERROr: " + str(err))
        print(label)
    except RuntimeWarning as warn:
        print("Warnnnn" + str(warn))
        print(label)


def init_metrics_for_LDA():
    return {'label': [], 'tol': [], 'solver': [], 'shrinkage': [], 'n_components': [],
            'store_covariance': [], 'covariance_estimator': [],
            'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': []
            }


def run_algorithm_lda_parallel(filename='', path='', stratify=False, train_size=0.8,
                               normalize_data=False, scaler='min-max', no_threads=8):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_LDA()
    my_filename = os.path.join(path, 'results/qlda', filename)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_LDA, header=True)

    tol_list = list(chain(np.random.uniform(low=1e-7, high=0.02, size=(1000,))))
    solver_list = ['lsqr', 'svd', 'eigen']
    shrinkage_list = list(chain(['auto', None], [0.5, 0.3, 0.4, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56,
                                                 0.57, 0.58, 59, 0.6, 0.4, 0.41, 0.42, 0.43, 0.44,
                                                 0.45, 0.46, 0.47, 0.48, 0.49, 0.2, 0.21, 0.22, 0.23, 0.24,
                                                 0.25, 0.26, 0.27, 0.28, 0.29, 0.31, 0.32, 0.33, 0.34, 0.35,
                                                 0.36, 0.37, 0.38, 0.39]))
    store_covariance_list = [True, False]
    n_components_list = [0, 1, 2, None]
    store_precision_list = [True, False]
    assume_centered_list = [True, False]

    # 5280000 configs

    with Manager() as manager:
        q_metrics = manager.Queue()
        jobs = []

        with Pool(no_threads) as pool:
            watcher = pool.apply_async(listener_write_to_file, (q_metrics, my_filename))
            for tol in tol_list:
                for solver in solver_list:
                    for shrinkage in shrinkage_list:
                        for store_covariance in store_covariance_list:
                            for n_components in n_components_list:
                                job = pool.apply_async(run_algorithm_lda_configuration_parallel,
                                                       (X, y, q_metrics, tol, solver, shrinkage, store_covariance,
                                                        n_components, None, stratify, train_size))
                                # print(job)
                                jobs.append(job)
                                for store_precision in store_precision_list:
                                    for assume_centered in assume_centered_list:
                                        job = pool.apply_async(run_algorithm_lda_configuration_parallel,
                                                               (X, y, q_metrics, tol, solver, shrinkage, store_covariance,
                                                                n_components,
                                                                EmpiricalCovariance(assume_centered=assume_centered,
                                                                                    store_precision=store_precision),
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


def run_algorithm_lda(filename='', path='', stratify=False, train_size=0.8,
                      normalize_data=False, scaler='min-max'):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_LDA()

    # full_path_filename = '/content/drive/MyDrive/code/' + filename
    # path_to_script = os.path.dirname(os.path.abspath(__file__))
    my_filename = os.path.join(path, 'results/qlda', filename)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_LDA, header=True)

    # default algorithm
    # run_algorithm_lda_configuration(metrics, 'LDA - default params', X, y, stratify=stratify, train_size=train_size,
    #                                 )
    #
    # metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_LDA)
    # # CALIBRATING tol
    # for tol in [1e-2, 1e-3, 1e-5, 1e-6, 1e-7]:
    #     run_algorithm_lda_configuration(metrics, 'LDA: tol = ' + str(tol), X, y, tol=tol,
    #                                     stratify=stratify, train_size=train_size)
    #
    # metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_LDA)
    #
    # # CALIBRATING solver
    # run_algorithm_lda_configuration(metrics, 'LDA: solver = lsqr', X, y, solver='lsqr', stratify=stratify,
    #                                 train_size=train_size)
    #
    # # CALIBRATING shrinkage
    # run_algorithm_lda_configuration(metrics, 'LDA: solver = lsqr, shrinkage = auto', X, y, solver='lsqr',
    #                                 shrinkage='auto', stratify=stratify, train_size=train_size)
    # run_algorithm_lda_configuration(metrics, 'LDA: solver = eigen, shrinkage = auto', X, y, solver='eigen',
    #                                 shrinkage='auto', stratify=stratify, train_size=train_size)
    # for shrinkage in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    #     run_algorithm_lda_configuration(metrics, 'LDA: solver = eigen, shrinkage = ' + str(shrinkage), X, y,
    #                                     solver='eigen', shrinkage=shrinkage, stratify=stratify, train_size=train_size,
    #                                     )
    #     run_algorithm_lda_configuration(metrics, 'LDA: solver = lsqr, shrinkage = ' + str(shrinkage), X, y,
    #                                     solver='lsqr', shrinkage=shrinkage, stratify=stratify, train_size=train_size,
    #                                     )
    #
    # # CALIBRATING n_components
    # run_algorithm_lda_configuration(metrics, 'LDA: n_components = 0', X, y, n_components=0, stratify=stratify,
    #                                 train_size=train_size)
    # run_algorithm_lda_configuration(metrics, 'LDA: n_components = 1', X, y, n_components=1, stratify=stratify,
    #                                 train_size=train_size)
    #
    # # CALIBRATING store_covariance
    # run_algorithm_lda_configuration(metrics, 'LDA: store_covariance = True, solver = svd', X, y, solver='svd',
    #                                 store_covariance=True, stratify=stratify, train_size=train_size)
    #
    # # CALIBRATING covariance_estimator
    # from sklearn.covariance import EmpiricalCovariance, EllipticEnvelope
    # from sklearn.covariance import GraphicalLassoCV
    #
    # # EmpiricalCovariance -------calibrating params
    # run_algorithm_lda_configuration(metrics, 'LDA: covariance_estimator = EmpiricalCovariance, solver = lsqr', X, y,
    #                                 solver='lsqr', covariance_estimator=EmpiricalCovariance(), stratify=stratify,
    #                                 train_size=train_size)
    # run_algorithm_lda_configuration(metrics,
    #                                 'LDA: covariance_estimator = EmpiricalCovariance(assume_centered = True), solver = lsqr',
    #                                 X, y, solver='lsqr', covariance_estimator=EmpiricalCovariance(assume_centered=True),
    #                                 stratify=stratify, train_size=train_size)
    #
    # # EllipticEnvelope ----- calibrating params as well
    # run_algorithm_lda_configuration(metrics,
    #                                 'LDA: covariance_estimator = EllipticEnvelope(random_state=0), solver = lsqr', X, y,
    #                                 solver='lsqr', covariance_estimator=EllipticEnvelope(random_state=0),
    #                                 stratify=stratify, train_size=train_size)
    # run_algorithm_lda_configuration(metrics,
    #                                 'LDA: covariance_estimator = EllipticEnvelope(random_state=0, assume_centered = True), solver = lsqr',
    #                                 X, y, solver='lsqr',
    #                                 covariance_estimator=EllipticEnvelope(random_state=0, assume_centered=True),
    #                                 stratify=stratify, train_size=train_size)
    #
    # for support_fraction in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    #     run_algorithm_lda_configuration(metrics,
    #                                     'LDA: covariance_estimator = EllipticEnvelope(random_state=0, support_fraction = ' + str(
    #                                         support_fraction) + '), solver = lsqr', X, y, solver='lsqr',
    #                                     covariance_estimator=EllipticEnvelope(random_state=0,
    #                                                                           support_fraction=support_fraction),
    #                                     stratify=stratify, train_size=train_size)
    #
    # for contamination in [0.1, 0.2, 0.3, 0.4, 0.5]:
    #     run_algorithm_lda_configuration(metrics,
    #                                     'LDA: covariance_estimator = EllipticEnvelope(random_state=0, contamination = ' + str(
    #                                         contamination) + '), solver = lsqr', X, y, solver='lsqr',
    #                                     covariance_estimator=EllipticEnvelope(random_state=0,
    #                                                                           contamination=contamination),
    #                                     stratify=stratify, train_size=train_size)
    #
    # # GraphicalLassoCV ----------- with params
    # run_algorithm_lda_configuration(metrics, 'LDA: covariance_estimator = GraphicalLassoCV, solver = lsqr', X, y,
    #                                 solver='lsqr',
    #                                 covariance_estimator=GraphicalLassoCV(), stratify=stratify, train_size=train_size,
    #                                 )
    #
    # for alphas in range(1, 4, 1):
    #     run_algorithm_lda_configuration(metrics, 'LDA: covariance_estimator = GraphicalLassoCV(n_refinements = ' + str(
    #         alphas) + '), solver = lsqr',
    #                                     X, y, solver='lsqr',
    #                                     covariance_estimator=GraphicalLassoCV(n_refinements=alphas),
    #                                     stratify=stratify, train_size=train_size)
    #
    # for alphas in range(5, 15, 1):
    #     run_algorithm_lda_configuration(metrics, 'LDA: covariance_estimator = GraphicalLassoCV(alphas = ' + str(
    #         alphas) + '), solver = lsqr',
    #                                     X, y, solver='lsqr', covariance_estimator=GraphicalLassoCV(alphas=alphas),
    #                                     stratify=stratify, train_size=train_size)
    #     run_algorithm_lda_configuration(metrics,
    #                                     'LDA: covariance_estimator = GraphicalLassoCV(mode = lars, alphas = ' + str(
    #                                         alphas) + '), solver = lsqr', X, y, solver='lsqr',
    #                                     covariance_estimator=GraphicalLassoCV(mode='lars', alphas=alphas),
    #                                     stratify=stratify, train_size=train_size)
    #     run_algorithm_lda_configuration(metrics,
    #                                     'LDA: covariance_estimator = GraphicalLassoCV(assume_centered = True, alphas = ' + str(
    #                                         alphas) + '), solver = lsqr', X, y, solver='lsqr',
    #                                     covariance_estimator=GraphicalLassoCV(assume_centered=True, alphas=alphas),
    #                                     stratify=stratify, train_size=train_size)
    #     run_algorithm_lda_configuration(metrics,
    #                                     'LDA: covariance_estimator = GraphicalLassoCV(mode = lars, assume_centered = True, alphas = ' + str(
    #                                         alphas) + '), solver = lsqr', X, y, solver='lsqr',
    #                                     covariance_estimator=GraphicalLassoCV(mode='lars', assume_centered=True,
    #                                                                           alphas=alphas), stratify=stratify,
    #                                     train_size=train_size)
    #
    #     run_algorithm_lda_configuration(metrics, 'LDA: covariance_estimator = GraphicalLassoCV(n_refinements = ' + str(
    #         alphas) + '), solver = lsqr',
    #                                     X, y, solver='lsqr',
    #                                     covariance_estimator=GraphicalLassoCV(n_refinements=alphas),
    #                                     stratify=stratify, train_size=train_size)
    #     run_algorithm_lda_configuration(metrics,
    #                                     'LDA: covariance_estimator = GraphicalLassoCV(mode = lars, n_refinements = ' + str(
    #                                         alphas) + '), solver = lsqr', X, y, solver='lsqr',
    #                                     covariance_estimator=GraphicalLassoCV(mode='lars', n_refinements=alphas),
    #                                     stratify=stratify, train_size=train_size)
    #     run_algorithm_lda_configuration(metrics,
    #                                     'LDA: covariance_estimator = GraphicalLassoCV(assume_centered = True, n_refinements = ' + str(
    #                                         alphas) + '), solver = lsqr', X, y, solver='lsqr',
    #                                     covariance_estimator=GraphicalLassoCV(assume_centered=True,
    #                                                                           n_refinements=alphas), stratify=stratify,
    #                                     train_size=train_size)
    #     run_algorithm_lda_configuration(metrics,
    #                                     'LDA: covariance_estimator = GraphicalLassoCV(mode = lars, assume_centered = True, n_refinements = ' + str(
    #                                         alphas) + '), solver = lsqr', X, y, solver='lsqr',
    #                                     covariance_estimator=GraphicalLassoCV(mode='lars', assume_centered=True,
    #                                                                           n_refinements=alphas), stratify=stratify,
    #                                     train_size=train_size)
    # metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_LDA)
    # for cv in range(2, 11, 1):
    #     run_algorithm_lda_configuration(metrics, 'LDA: covariance_estimator = GraphicalLassoCV(cv = ' + str(
    #         cv) + '), solver = lsqr',
    #                                     X, y, solver='lsqr', covariance_estimator=GraphicalLassoCV(cv=cv),
    #                                     stratify=stratify, train_size=train_size)
    #     run_algorithm_lda_configuration(metrics,
    #                                     'LDA: covariance_estimator = GraphicalLassoCV(mode = lars, cv = ' + str(
    #                                         cv) + '), solver = lsqr', X, y, solver='lsqr',
    #                                     covariance_estimator=GraphicalLassoCV(mode='lars', cv=cv), stratify=stratify,
    #                                     train_size=train_size)
    #     run_algorithm_lda_configuration(metrics,
    #                                     'LDA: covariance_estimator = GraphicalLassoCV(assume_centered = True, cv = ' + str(
    #                                         cv) + '), solver = lsqr', X, y, solver='lsqr',
    #                                     covariance_estimator=GraphicalLassoCV(assume_centered=True, cv=cv),
    #                                     stratify=stratify, train_size=train_size)
    #     run_algorithm_lda_configuration(metrics,
    #                                     'LDA: covariance_estimator = GraphicalLassoCV(mode = lars, assume_centered = True, cv = ' + str(
    #                                         cv) + '), solver = lsqr', X, y, solver='lsqr',
    #                                     covariance_estimator=GraphicalLassoCV(mode='lars', assume_centered=True, cv=cv),
    #                                     stratify=stratify, train_size=train_size)
    #
    # for tol in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
    #     run_algorithm_lda_configuration(metrics, 'LDA: covariance_estimator = GraphicalLassoCV(tol = ' + str(
    #         tol) + '), solver = lsqr',
    #                                     X, y, solver='lsqr', covariance_estimator=GraphicalLassoCV(tol=tol),
    #                                     stratify=stratify, train_size=train_size)
    #     run_algorithm_lda_configuration(metrics,
    #                                     'LDA: covariance_estimator = GraphicalLassoCV(mode = lars, tol = ' + str(
    #                                         tol) + '), solver = lsqr', X, y, solver='lsqr',
    #                                     covariance_estimator=GraphicalLassoCV(mode='lars', tol=tol), stratify=stratify,
    #                                     train_size=train_size)
    #     run_algorithm_lda_configuration(metrics,
    #                                     'LDA: covariance_estimator = GraphicalLassoCV(assume_centered = True, tol = ' + str(
    #                                         tol) + '), solver = lsqr', X, y, solver='lsqr',
    #                                     covariance_estimator=GraphicalLassoCV(assume_centered=True, tol=tol),
    #                                     stratify=stratify, train_size=train_size)
    #     run_algorithm_lda_configuration(metrics,
    #                                     'LDA: covariance_estimator = GraphicalLassoCV(mode = lars, assume_centered = True, tol = ' + str(
    #                                         tol) + '), solver = lsqr', X, y, solver='lsqr',
    #                                     covariance_estimator=GraphicalLassoCV(mode='lars', assume_centered=True,
    #                                                                           tol=tol), stratify=stratify,
    #                                     train_size=train_size)
    #     run_algorithm_lda_configuration(metrics, 'LDA: covariance_estimator = GraphicalLassoCV(enet_tol = ' + str(
    #         tol) + '), solver = lsqr',
    #                                     X, y, solver='lsqr', covariance_estimator=GraphicalLassoCV(enet_tol=tol),
    #                                     stratify=stratify, train_size=train_size)
    #     run_algorithm_lda_configuration(metrics,
    #                                     'LDA: covariance_estimator = GraphicalLassoCV(mode = lars, enet_tol = ' + str(
    #                                         tol) + '), solver = lsqr', X, y, solver='lsqr',
    #                                     covariance_estimator=GraphicalLassoCV(mode='lars', enet_tol=tol),
    #                                     stratify=stratify, train_size=train_size,)
    #     run_algorithm_lda_configuration(metrics,
    #                                     'LDA: covariance_estimator = GraphicalLassoCV(assume_centered = True, enet_tol = ' + str(
    #                                         tol) + '), solver = lsqr', X, y, solver='lsqr',
    #                                     covariance_estimator=GraphicalLassoCV(assume_centered=True, enet_tol=tol),
    #                                     stratify=stratify, train_size=train_size, )
    #     run_algorithm_lda_configuration(metrics,
    #                                     'LDA: covariance_estimator = GraphicalLassoCV(mode = lars, assume_centered = True, enet_tol = ' + str(
    #                                         tol) + '), solver = lsqr', X, y, solver='lsqr',
    #                                     covariance_estimator=GraphicalLassoCV(mode='lars', assume_centered=True,
    #                                                                           enet_tol=tol), stratify=stratify,
    #                                     train_size=train_size)
    #
    # for max_iter in range(10, 100, 10):
    #     run_algorithm_lda_configuration(metrics, 'LDA: covariance_estimator = GraphicalLassoCV(max_iter = ' + str(
    #         max_iter) + '), solver = lsqr',
    #                                     X, y, solver='lsqr', covariance_estimator=GraphicalLassoCV(max_iter=max_iter),
    #                                     stratify=stratify, train_size=train_size)
    #     run_algorithm_lda_configuration(metrics,
    #                                     'LDA: covariance_estimator = GraphicalLassoCV(mode = lars, max_iter = ' + str(
    #                                         max_iter) + '), solver = lsqr', X, y, solver='lsqr',
    #                                     covariance_estimator=GraphicalLassoCV(mode='lars', max_iter=max_iter),
    #                                     stratify=stratify, train_size=train_size)
    #     run_algorithm_lda_configuration(metrics,
    #                                     'LDA: covariance_estimator = GraphicalLassoCV(assume_centered = True, max_iter = ' + str(
    #                                         max_iter) + '), solver = lsqr', X, y, solver='lsqr',
    #                                     covariance_estimator=GraphicalLassoCV(assume_centered=True, max_iter=max_iter),
    #                                     stratify=stratify, train_size=train_size)
    #     run_algorithm_lda_configuration(metrics,
    #                                     'LDA: covariance_estimator = GraphicalLassoCV(mode = lars, assume_centered = True, max_iter = ' + str(
    #                                         max_iter) + '), solver = lsqr', X, y, solver='lsqr',
    #                                     covariance_estimator=GraphicalLassoCV(mode='lars', assume_centered=True,
    #                                                                           max_iter=max_iter), stratify=stratify,
    #                                     train_size=train_size )
    #
    #     run_algorithm_lda_configuration(metrics, 'LDA: covariance_estimator = GraphicalLassoCV(max_iter = ' + str(
    #         max_iter + 100) + '), solver = lsqr',
    #                                     X, y, solver='lsqr',
    #                                     covariance_estimator=GraphicalLassoCV(max_iter=max_iter + 100),
    #                                     stratify=stratify, train_size=train_size)
    #     run_algorithm_lda_configuration(metrics,
    #                                     'LDA: covariance_estimator = GraphicalLassoCV(mode = lars, max_iter = ' + str(
    #                                         max_iter + 100) + '), solver = lsqr', X, y, solver='lsqr',
    #                                     covariance_estimator=GraphicalLassoCV(mode='lars', max_iter=max_iter + 100),
    #                                     stratify=stratify, train_size=train_size,)
    #     run_algorithm_lda_configuration(metrics,
    #                                     'LDA: covariance_estimator = GraphicalLassoCV(assume_centered = True, max_iter = ' + str(
    #                                         max_iter + 100) + '), solver = lsqr', X, y, solver='lsqr',
    #                                     covariance_estimator=GraphicalLassoCV(assume_centered=True,
    #                                                                           max_iter=max_iter + 100),
    #                                     stratify=stratify, train_size=train_size, )
    #     run_algorithm_lda_configuration(metrics,
    #                                     'LDA: covariance_estimator = GraphicalLassoCV(mode = lars, assume_centered = True, max_iter = ' + str(
    #                                         max_iter + 100) + '), solver = lsqr', X, y, solver='lsqr',
    #                                     covariance_estimator=GraphicalLassoCV(mode='lars', assume_centered=True,
    #                                                                           max_iter=max_iter + 100),
    #                                     stratify=stratify, train_size=train_size)

    # metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_LDA)

    for tol in np.random.uniform(low=1e-7, high=0.02, size=(1000,)):
        for solver in ['lsqr', 'svd', 'eigen']:
            for shrinkage in chain(['auto', None], [0.5, 0.3, 0.4, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56,
                                                    0.57, 0.58, 59, 0.6, 0.4, 0.41, 0.42, 0.43, 0.44,
                                                    0.45, 0.46, 0.47, 0.48, 0.49, 0.2, 0.21, 0.22, 0.23, 0.24,
                                                    0.25, 0.26, 0.27, 0.28, 0.29, 0.31, 0.32, 0.33, 0.34, 0.35,
                                                    0.36, 0.37, 0.38, 0.39]):
                for store_covariance in [True, False]:
                    for n_components in [0, 1, 2, None]:
                        try:
                            run_algorithm_lda_configuration(metrics, 'LDA: tol = ' + str(tol) + ', solver' + solver +
                                                            ', shrinkage= ' + str(
                                shrinkage) + ', store_covariance=' + str(
                                store_covariance) + ', n_components' + str(
                                n_components) + ', covariance_estimator = None',
                                                            X, y, tol=tol, solver=solver,
                                                            shrinkage=shrinkage, store_covariance=store_covariance,
                                                            n_components=n_components, covariance_estimator=None,
                                                            stratify=stratify, train_size=train_size)
                            for store_precision in [True, False]:
                                for assume_centered in [True, False]:
                                    run_algorithm_lda_configuration(metrics,
                                                                    'LDA: tol = ' + str(tol) + ', solver' + solver +
                                                                    ', shrinkage= ' + str(
                                                                        shrinkage) + ', store_covariance=' + str(
                                                                        store_covariance)
                                                                    + ', n_components' + str(n_components) +
                                                                    ', covariance_estimator = EmpiricalCovariance(store_precision=' + str(
                                                                        store_precision) + ', assume_centered=' + str(
                                                                        assume_centered) + ')',
                                                                    X, y, tol=tol, solver=solver,
                                                                    shrinkage=shrinkage,
                                                                    store_covariance=store_covariance,
                                                                    n_components=n_components,
                                                                    covariance_estimator=EmpiricalCovariance(
                                                                        assume_centered=assume_centered,
                                                                        store_precision=store_precision),
                                                                    stratify=stratify, train_size=train_size)
                        except Exception as err:
                            print(err)
                        metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_LDA)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_LDA)


def run_best_configs_lda(df_configs, filename='', path='', stratify=True, train_size=0.8,
                         normalize_data=True, scaler='min-max', n_rep=100):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_LDA()
    my_filename = os.path.join(path, 'new_results/qlda', filename)

    for index, row in df_configs.iterrows():
        label = create_label_LDA(row['tol'], row['solver'], row['shrinkage'], row['store_covariance'],
                                 row['n_components'], row['covariance_estimator'])
        if (row['shrinkage'] == 'None' or row['shrinkage'] == None or str(row['shrinkage']) == 'nan'):
            shrinkage = None
        elif (row['shrinkage'] == 'auto'):
            shrinkage = 'auto'
        else:
            shrinkage = float(row['shrinkage'])

        if (row['n_components'] == None or row['n_components'] == 'None' or str(row['n_components']) == 'nan'):
            n_components = None
        else:
            n_components = int(row['n_components'])

        if (row['covariance_estimator'] == "EmpiricalCovariance(store_precision=True, assume_centered=False)"):
            covariance_estimator = EmpiricalCovariance(store_precision=True, assume_centered=False)
        elif (row['covariance_estimator'] == "EmpiricalCovariance(store_precision=False, assume_centered=False)"):
            covariance_estimator = EmpiricalCovariance(store_precision=False, assume_centered=False)
        else:
            covariance_estimator = None
        for i in range(0, n_rep):
            run_algorithm_lda_configuration(metrics, label, X, y, tol=float(row['tol']),
                                            solver=row['solver'], shrinkage=shrinkage,
                                            store_covariance=bool(row['store_covariance']),
                                            n_components=n_components, covariance_estimator=covariance_estimator,
                                            stratify=stratify, train_size=train_size
                                            )

    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.groupby(['label'], as_index=False).agg({'precision': 'mean', 'recall': 'mean',
                                                                    'f1_score': 'mean', 'roc_auc': 'mean',
                                                                    'tol': 'first',
                                                                    'solver': 'first',
                                                                    'shrinkage': 'first',
                                                                    'store_covariance': 'first',
                                                                    'n_components': 'first',
                                                                    'covariance_estimator': 'first'})
    metrics_df = compute_average_metric(metrics_df)
    metrics_df.sort_values(by=['average_metric'], ascending=False, inplace=True)
    metrics = appendMetricsTOCSV(my_filename, metrics_df, init_metrics_for_LDA, header=True)


def create_label_LDA_for_row(row):
    return create_label_LDA(row['tol'], row['solver'], row['shrinkage'], row['store_covariance'],
                            row['n_components'], row['covariance_estimator'])


def create_label_LDA(tol, solver, shrinkage, store_covariance, n_components, covariance_estimator):
    label = "Linear Discriminant Analysis, tol=" + str(tol) + ", solver=" + str(solver) + \
            ', shrinkage=' + str(shrinkage) + ', store_covariance=' + str(store_covariance) + \
            ", n_components=" + str(n_components) + ", covariance_estimator=" + str(covariance_estimator)
    return label
