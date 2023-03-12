import os
from itertools import chain

import numpy as np
from sklearn.naive_bayes import GaussianNB, BernoulliNB

from data_pre import split_data_in_testing_training, load_normalized_dataset
from utils import prediction, cal_metrics, appendMetricsTOCSV


def run_algorithm_gnb_configuration(metrics, label, X, y,
                                    var_smoothing=1e-9,
                                    stratify=False, train_size=0.8):
    X_test, X_train, y_test, y_train = split_data_in_testing_training(X, y, stratify, train_size)
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


def run_algorithm_bnb_configuration(metrics, label, X, y,
                                    alpha=1.0, binarize=0.0, fit_prior=True,
                                    stratify=False, train_size=0.8):
    X_test, X_train, y_test, y_train = split_data_in_testing_training(X, y, stratify, train_size)

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
