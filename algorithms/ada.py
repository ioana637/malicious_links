import os
from itertools import chain
import warnings

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier

from data_post import compute_average_metric
from data_pre import split_data_in_testing_training, load_normalized_dataset

warnings.filterwarnings("error")

import numpy as np

from utils import split_data, prediction, cal_metrics, appendMetricsTOCSV


def run_algorithm_ada_boost_configuration(metrics, label, X, y,
                                          base_estimator=None,
                                          n_estimators=50,
                                          learning_rate=1.0,
                                          algorithm='SAMME.R',
                                          stratify=False, train_size=0.8):
    X_test, X_train, y_test, y_train = split_data_in_testing_training(X, y, stratify, train_size)

    try:
        # Creating the classifier object
        classifier = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators,
                                        learning_rate=learning_rate, algorithm=algorithm)
        # Performing training
        classifier.fit(X_train, y_train)

        # Make predictions
        y_pred, y_pred_probabilities = prediction(X_test, classifier)

        # Compute metrics
        precision, recall, f1, roc_auc = cal_metrics(y_test, y_pred, y_pred_probabilities, label)
        metrics['label'].append(label)

        metrics['base_estimator'].append(base_estimator)
        metrics['n_estimators'].append(n_estimators)
        metrics['learning_rate'].append(learning_rate)
        metrics['algorithm'].append(algorithm)

        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1_score'].append(f1)
        metrics['roc_auc'].append(roc_auc)
    except Exception as er:
        # print(er)
        pass
    except RuntimeWarning as warn:
        # print(warn)
        pass


def init_metrics_for_AdaBoost():
    return {'label': [],
            'base_estimator': [], 'n_estimators': [],
            'learning_rate': [], 'algorithm': [],
            'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': []
            }


def run_algorithm_ada_boost(filename='', path='', stratify=False, train_size=0.8,
                            normalize_data=False, scaler='min-max'):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_AdaBoost()

    # full_path_filename = '/content/drive/MyDrive/code/' + filename
    # path_to_script = os.path.dirname(os.path.abspath(__file__))
    my_filename = os.path.join(path, 'results/ada', filename)

    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_AdaBoost, header=True)

    # default algorithm
    # run_algorithm_ada_boost_configuration(metrics, 'Ada Boost - default params', X, y, stratify=stratify,
    #                                       train_size=train_size)
    # run_algorithm_ada_boost_configuration(metrics, 'Ada Boost - algorithm = SAMME', X, y, algorithm='SAMME',
    #                                       stratify=stratify, train_size=train_size)
    # metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_AdaBoost)
    # # CALIBRATING n_estimators
    # for n_estimators in range(1, 100, 2):
    #     run_algorithm_ada_boost_configuration(metrics, 'Ada Boost - n_estimators = ' + str(n_estimators), X, y,
    #                                           n_estimators=n_estimators, stratify=stratify,
    #                                           train_size=train_size)
    #     run_algorithm_ada_boost_configuration(metrics, 'Ada Boost - algorithm = SAMME, n_estimators = ' +
    #                                           str(n_estimators), X, y, algorithm='SAMME',
    #                                           n_estimators=n_estimators, stratify=stratify,
    #                                           train_size=train_size)
    # for n_estimators in range(100, 200, 5):
    #     run_algorithm_ada_boost_configuration(metrics, 'Ada Boost - n_estimators = ' + str(n_estimators), X, y,
    #                                           n_estimators=n_estimators, stratify=stratify,
    #                                           train_size=train_size)
    #     run_algorithm_ada_boost_configuration(metrics, 'Ada Boost - algorithm = SAMME, n_estimators = ' +
    #                                           str(n_estimators), X, y, algorithm='SAMME',
    #                                           n_estimators=n_estimators, stratify=stratify,
    #                                           train_size=train_size)
    # for n_estimators in range(200, 300, 10):
    #     run_algorithm_ada_boost_configuration(metrics, 'Ada Boost - n_estimators = ' + str(n_estimators), X, y,
    #                                           n_estimators=n_estimators, stratify=stratify,
    #                                           train_size=train_size)
    #     run_algorithm_ada_boost_configuration(metrics, 'Ada Boost - algorithm = SAMME, n_estimators = ' +
    #                                           str(n_estimators), X, y, algorithm='SAMME',
    #                                           n_estimators=n_estimators, stratify=stratify,
    #                                           train_size=train_size)
    # metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_AdaBoost)
    #
    # for n_estimators in range(300, 500, 25):
    #     run_algorithm_ada_boost_configuration(metrics, 'Ada Boost - n_estimators = ' + str(n_estimators), X, y,
    #                                           n_estimators=n_estimators, stratify=stratify,
    #                                           train_size=train_size)
    #     run_algorithm_ada_boost_configuration(metrics, 'Ada Boost -  algorithm = SAMME, n_estimators = ' +
    #                                           str(n_estimators), X, y, algorithm='SAMME',
    #                                           n_estimators=n_estimators, stratify=stratify,
    #                                           train_size=train_size)
    # for n_estimators in range(500, 1000, 50):
    #     run_algorithm_ada_boost_configuration(metrics, 'Ada Boost - n_estimators = ' + str(n_estimators), X, y,
    #                                           n_estimators=n_estimators, stratify=stratify,
    #                                           train_size=train_size)
    #     run_algorithm_ada_boost_configuration(metrics, 'Ada Boost - algorithm = SAMME, n_estimators = ' +
    #                                           str(n_estimators), X, y, algorithm='SAMME',
    #                                           n_estimators=n_estimators, stratify=stratify,
    #                                           train_size=train_size)
    #
    #     # CALIBRATING learning_rate
    # for learning_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3,
    #                       1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
    #                       ]:
    #     run_algorithm_ada_boost_configuration(metrics, 'Ada Boost - learning_rate = ' + str(learning_rate),
    #                                           X, y, learning_rate=learning_rate, stratify=stratify,
    #                                           train_size=train_size)
    #     run_algorithm_ada_boost_configuration(metrics, 'Ada Boost - algorithm = SAMME, learning_rate = ' +
    #                                           str(learning_rate), X, y, algorithm='SAMME',
    #                                           learning_rate=learning_rate, stratify=stratify,
    #                                           train_size=train_size)

    for algorithm in ['SAMME', 'SAMME.R']:
        for learning_rate in np.random.uniform(low=0.7, high=1.2, size=(20,)):
            for n_estimators in chain(range(31, 160, 1), range(175, 195, 1), range(340, 360, 1), range(645, 655, 1)):
                run_algorithm_ada_boost_configuration(metrics, 'Ada Boost - learning_rate = ' + str(learning_rate) +
                                                      ', algorithm=' + algorithm + ', n_estimators=' +
                                                      str(n_estimators), X, y, learning_rate=learning_rate,
                                                      algorithm=algorithm, n_estimators=n_estimators,
                                                      stratify=stratify, train_size=train_size)
            metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_AdaBoost)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_AdaBoost)


def run_best_configs_ada(df_configs, filename='', path='', stratify=True, train_size=0.8,
                     normalize_data=True, scaler='min-max', n_rep=100):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_AdaBoost()
    my_filename = os.path.join(path, 'new_results/ada', filename)

    for i in range(1, n_rep):
        for index, row in df_configs.iterrows():
            label = "ADA Boost, base estimator=None, n_estimators=" + str(
                row['n_estimators']) + ", learning_rate=" + str(row['learning_rate']) + ", algorithm=" + row[
                        'algorithm']
            run_algorithm_ada_boost_configuration(metrics, label, X, y,
                                                  base_estimator=None,
                                                  n_estimators=int(row['n_estimators']),
                                                  learning_rate=float(row['learning_rate']),
                                                  algorithm=row['algorithm'],
                                                  stratify=stratify, train_size=train_size)

    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.groupby(['label'], as_index=False).agg({'precision': 'mean', 'recall': 'mean',
                                                                    'f1_score': 'mean', 'roc_auc': 'mean',
                                                                    'base_estimator': 'first',
                                                                    'n_estimators': 'first',
                                                                    'learning_rate': 'first',
                                                                    'algorithm': 'first'})
    metrics_df = compute_average_metric(metrics_df)
    metrics_df.sort_values(by=['average_metric'], ascending=False, inplace=True)
    metrics = appendMetricsTOCSV(my_filename, metrics_df, init_metrics_for_AdaBoost, header=True)
