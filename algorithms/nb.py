import os
from itertools import chain
from multiprocessing import Manager, Pool

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, BernoulliNB

from data_post import compute_average_metric
from data_pre import split_data_in_testing_training, load_normalized_dataset
from utils import prediction, cal_metrics, appendMetricsTOCSV, listener_write_to_file, convert_metrics_to_csv



def run_algorithm_gnb_configuration_parallel(X, y, q_metrics,
                                             var_smoothing=1e-9,
                                             stratify=False, train_size=0.8
                                             ):
    X_train, X_test, y_train, y_test = split_data_in_testing_training(X, y, stratify, train_size)
    try:
        # Creating the classifier object
        classifier = GaussianNB(var_smoothing=var_smoothing)
        # Performing training
        classifier.fit(X_train, y_train)

        # Make predictions
        y_pred, y_pred_probabilities = prediction(X_test, classifier)

        # Compute metrics
        label = create_label_GNB(var_smoothing)
        precision, recall, f1, roc_auc = cal_metrics(y_test, y_pred, y_pred_probabilities, label, classifier)
        string_results_for_queue = convert_metrics_to_csv(',', label, var_smoothing,
                                                          precision, recall, f1, roc_auc)
        q_metrics.put(string_results_for_queue)
    except Exception as er:
        # pass
        print(er)
        # traceback.print_exc()
        # print(traceback.format_exc())
    except RuntimeWarning as warn:
        # pass
        print(warn)

def run_algorithm_gnb_configuration(metrics, label, X, y,
                                    var_smoothing=1e-9,
                                    stratify=False, train_size=0.8):
    X_train, X_test, y_train, y_test = split_data_in_testing_training(X, y, stratify, train_size)
    try:
        # Creating the classifier object
        classifier = GaussianNB(var_smoothing=var_smoothing)

        # Performing training
        classifier.fit(X_train, y_train)

        # Make predictions
        y_pred, y_pred_probabilities = prediction(X_test, classifier)

        # Compute metrics
        precision, recall, f1, roc_auc = cal_metrics(y_test, y_pred, y_pred_probabilities, label)
        metrics['label'].append(label)

        metrics['var_smoothing'].append(var_smoothing)

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


def init_metrics_for_GNB():
    return {'label': [],
            'var_smoothing': [],
            'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': []
            }


def run_algorithm_gnb_parallel(filename='', path='', stratify=False, train_size=0.8,
                               normalize_data=False, scaler='min-max', no_threads=8):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_GNB()
    my_filename = os.path.join(path, 'results/nb', filename)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_GNB, header=True)

    var_smoothing_list = list(chain(np.random.uniform(low=1e-10, high=1e-9, size=(100,)),
                                    np.random.uniform(low=1e-9, high=1e-8, size=(100,))))
    # 200

    with Manager() as manager:
        q_metrics = manager.Queue()
        jobs = []

        with Pool(no_threads) as pool:
            watcher = pool.apply_async(listener_write_to_file, (q_metrics, my_filename))
            for var_smoothing in var_smoothing_list:
                job = pool.apply_async(run_algorithm_gnb_configuration_parallel,
                                       (X, y, q_metrics, var_smoothing, stratify, train_size))
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


def run_algorithm_gnb(filename='', path='', stratify=False, train_size=0.8,
                      normalize_data=False, scaler='min-max'):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_GNB()

    # full_path_filename = '/content/drive/MyDrive/code/' + filename
    # path_to_script = os.path.dirname(os.path.abspath(__file__))
    my_filename = os.path.join(path, 'results/nb', filename)

    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_GNB, header=True)
    # default algorithm
    # run_algorithm_gnb_configuration(metrics, 'Gaussian NB - default params', X, y,
    #                                 stratify=stratify, train_size=train_size)
    #
    # # CALIBRATING var_smoothing
    # for var_smoothing in [1e-7, 1e-8, 1e-10, 1e-11, 1e-12]:
    #     run_algorithm_gnb_configuration(metrics, 'Gaussian NB - var_smoothing = ' + str(var_smoothing),
    #                                     X, y, var_smoothing=var_smoothing, stratify=stratify,
    #                                     train_size=train_size)
    # for var_smoothing in chain(np.random.uniform(low=1e-10, high=1e-9, size=(50,)),
    #                            np.random.uniform(low=1e-9, high=1e-8, size=(50,))):
    #     run_algorithm_gnb_configuration(metrics, 'Gaussian NB - var_smoothing = ' + str(var_smoothing),
    #                                     X, y, var_smoothing=var_smoothing, stratify=stratify,
    #                                     train_size=train_size)
    for var_smoothing in chain(np.random.uniform(low=1e-10, high=1e-9, size=(100,)),
                               np.random.uniform(low=1e-9, high=1e-8, size=(100,))):
        run_algorithm_gnb_configuration(metrics, 'Gaussian NB - var_smoothing = ' + str(var_smoothing),
                                        X, y, var_smoothing=var_smoothing, stratify=stratify,
                                        train_size=train_size)
        metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_GNB)


def init_metrics_for_BNB():
    return {'label': [],
            'alpha': [], 'fit_prior': [], 'binarize': [],
            'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': []
            }


def run_algorithm_bnb_parallel(filename='', path='', stratify=False, train_size=0.8,
                              normalize_data=False, scaler='min-max', no_threads=8):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_BNB()
    my_filename = os.path.join(path, 'results/nb', filename)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_BNB, header=True)

    alpha_list = list(np.random.uniform(low=0.0, high=2.5, size=(200,)))
    binarize_list = list(np.random.uniform(low=0.1, high=1.5, size=(100,)))
    fit_prior_list= [True, False]
    # 40000

    with Manager() as manager:
        q_metrics = manager.Queue()
        jobs = []

        with Pool(no_threads) as pool:
            watcher = pool.apply_async(listener_write_to_file, (q_metrics, my_filename))
            for alpha in alpha_list:
                for binarize in binarize_list:
                    for fit_prior in fit_prior_list:
                        job = pool.apply_async(run_algorithm_bnb_configuration_parallel,
                                                   (X, y, q_metrics,
                                                    alpha, binarize, fit_prior,
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


def run_algorithm_bnb(filename='', path='', stratify=False, train_size=0.8,
                      normalize_data=False, scaler='min-max'):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_BNB()

    # full_path_filename = '/content/drive/MyDrive/code/' + filename
    # path_to_script = os.path.dirname(os.path.abspath(__file__))
    my_filename = os.path.join(path, 'results/nb', filename)

    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_BNB, header=True)

    # default algorithm
    # run_algorithm_bnb_configuration(metrics, 'Bernoulli NB - default params', X, y,
    #                                 stratify=stratify, train_size=train_size)
    #
    # # CALIBRATING alpha
    # for alpha in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]:
    #     run_algorithm_bnb_configuration(metrics, 'Bernoulli NB - fit_prior = True, alpha = ' +
    #                                     str(alpha), X, y, alpha=alpha, stratify=stratify,
    #                                     train_size=train_size)
    #     # CALIBRATING fit_prior
    #     run_algorithm_bnb_configuration(metrics, 'Bernoulli NB - fit_prior = False, alpha = '
    #                                     + str(alpha), X, y, alpha=alpha, fit_prior=False,
    #                                     stratify=stratify, train_size=train_size)
    #     # CALIBRATING binarize
    #     for binarize in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0]:
    #         run_algorithm_bnb_configuration(metrics, 'Bernoulli NB - fit_prior = True, alpha = '
    #                                         + str(alpha) + ' binarize = ' + str(binarize), X, y,
    #                                         alpha=alpha, fit_prior=True, binarize=binarize,
    #                                         stratify=stratify, train_size=train_size)
    #         run_algorithm_bnb_configuration(metrics, 'Bernoulli NB - fit_prior = False, alpha = '
    #                                         + str(alpha) + ' binarize = ' + str(binarize), X, y,
    #                                         alpha=alpha, fit_prior=False, binarize=binarize,
    #                                         stratify=stratify, train_size=train_size)
    # metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_BNB)
    for alpha in np.random.uniform(low=0.0, high=2.5, size=(200,)):
        for binarize in np.random.uniform(low=0.1, high=1.5, size=(100,)):
            for fit_prior in [True, False]:
                run_algorithm_bnb_configuration(metrics, 'Bernoulli NB - fit_prior = ' + str(fit_prior) + ', alpha = '
                                                + str(alpha) + ' binarize = ' + str(binarize),
                                                X, y, alpha=alpha, fit_prior=fit_prior,
                                                binarize=binarize, stratify=stratify,
                                                train_size=train_size)
        metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_BNB)


def run_algorithm_bnb_configuration_parallel(X, y, q_metrics,
                                             alpha=1.0, binarize=0.0, fit_prior=True,
                                            stratify=False, train_size=0.8
                                            ):
    X_train, X_test, y_train, y_test = split_data_in_testing_training(X, y, stratify, train_size)
    try:
        # Creating the classifier object
        classifier = BernoulliNB(alpha=alpha, fit_prior=fit_prior, binarize=binarize)
        # Performing training
        classifier.fit(X_train, y_train)

        # Make predictions
        y_pred, y_pred_probabilities = prediction(X_test, classifier)

        # Compute metrics
        label = create_label_BNB(alpha=alpha, fit_prior=fit_prior, binarize=binarize)
        precision, recall, f1, roc_auc = cal_metrics(y_test, y_pred, y_pred_probabilities, label, classifier)
        string_results_for_queue = convert_metrics_to_csv(',', label, alpha, fit_prior, binarize,
                                                          precision, recall, f1, roc_auc)
        q_metrics.put(string_results_for_queue)
    except Exception as er:
        # pass
        print(er)
        # traceback.print_exc()
        # print(traceback.format_exc())
    except RuntimeWarning as warn:
        # pass
        print(warn)

def run_algorithm_bnb_configuration(metrics, label, X, y,
                                    alpha=1.0, binarize=0.0, fit_prior=True,
                                    stratify=False, train_size=0.8):
    X_train, X_test, y_train, y_test = split_data_in_testing_training(X, y, stratify, train_size)

    try:
        # Creating the classifier object
        classifier = BernoulliNB(alpha=alpha, fit_prior=fit_prior, binarize=binarize)

        # Performing training
        classifier.fit(X_train, y_train)

        # Make predictions
        y_pred, y_pred_probabilities = prediction(X_test, classifier)

        # Compute metrics
        precision, recall, f1, roc_auc = cal_metrics(y_test, y_pred, y_pred_probabilities, label)
        metrics['label'].append(label)

        metrics['alpha'].append(alpha)
        metrics['fit_prior'].append(fit_prior)
        metrics['binarize'].append(binarize)

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


def run_best_configs_gnb(df_configs, filename='', path='', stratify=True, train_size=0.8,
                         normalize_data=True, scaler='min-max', n_rep=100):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_GNB()
    my_filename = os.path.join(path, 'new_results/nb', filename)

    for i in range(1, n_rep):
        for index, row in df_configs.iterrows():
            label = create_label_GNB(row['var_smoothing'])
            run_algorithm_gnb_configuration(metrics, label, X, y,
                                            var_smoothing=float(row['var_smoothing']),
                                            stratify=stratify, train_size=train_size)

    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.groupby(['label'], as_index=False).agg({'precision': 'mean', 'recall': 'mean',
                                                                    'f1_score': 'mean', 'roc_auc': 'mean',
                                                                    'var_smoothing': 'first'})
    metrics_df = compute_average_metric(metrics_df)
    metrics_df.sort_values(by=['average_metric'], ascending=False, inplace=True)
    metrics = appendMetricsTOCSV(my_filename, metrics_df, init_metrics_for_GNB, header=True)

def create_label_GNB(var_smoothing):
    label = "Gaussian Naive Bayes, var_smoothing=" + str(var_smoothing)
    return label


def run_best_configs_bnb(df_configs, filename='', path='', stratify=True, train_size=0.8,
                        normalize_data=True, scaler='min-max', n_rep=100):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_BNB()
    my_filename = os.path.join(path, 'new_results/nb', filename)

    for i in range(1, n_rep):
        for index, row in df_configs.iterrows():
            label = create_label_BNB(row['alpha'], row['binarize'], row['fit_prior'])
            run_algorithm_bnb_configuration(metrics, label, X, y,
                                            alpha=float(row['alpha']), binarize=float(row['binarize']),
                                            fit_prior=bool(row['fit_prior']),
                                            stratify=stratify, train_size=train_size)

    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.groupby(['label'], as_index=False).agg({'precision': 'mean', 'recall': 'mean',
                                                                    'f1_score': 'mean', 'roc_auc': 'mean',
                                                                    'alpha': 'first',
                                                                    'binarize': 'first',
                                                                    'fit_prior': 'first'})
    metrics_df = compute_average_metric(metrics_df)
    metrics_df.sort_values(by=['average_metric'], ascending=False, inplace=True)
    metrics = appendMetricsTOCSV(my_filename, metrics_df, init_metrics_for_BNB, header=True)

def create_label_BNB(alpha,binarize, fit_prior):
    label = "Bernoulli Naive Bayes, alpha=" + str(
        alpha) + ", binarize=" + str(binarize) + ", fit_prior=" + str(fit_prior)
    return label