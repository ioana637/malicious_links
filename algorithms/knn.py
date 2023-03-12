import os

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from data_post import compute_average_metric
from data_pre import load_normalized_dataset, split_data_in_testing_training
from utils import prediction, split_data, cal_metrics, appendMetricsTOCSV

import numpy as np
from itertools import chain


def run_top_20_KNN_configs(filename='', path = '', stratify=False, train_size=0.8,
                           normalize_data=True, scaler='min-max', n_rep=100):
    y, X = load_normalized_dataset(file = None, normalize = normalize_data, scaler=scaler)
    metrics = init_metrics_for_KNN()

    # full_path_filename = '/content/drive/MyDrive/code/' + filename
    # path_to_script = os.path.dirname(os.path.abspath(__file__))
    my_filename = os.path.join(path, 'new_results/knn', filename)

    for i in range(1, n_rep):
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance,  metric = manhattan, p=7, algorithm = ball_tree, n_neighbors = 6, leaf size= 90'
                                        , X, y, weights='distance', metric='manhattan', p=7, algorithm='ball_tree',
                                        n_neighbors=6, leaf_size=90, stratify=stratify, train_size=train_size)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance,  metric = manhattan, p=2, algorithm = ball_tree, n_neighbors = 5, leaf size= 13'
                                        , X, y, weights='distance', metric='manhattan', p=2, algorithm='ball_tree',
                                        n_neighbors=5, leaf_size=13, stratify=stratify, train_size=train_size)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance,  metric = minkowski, p=4, algorithm = kd_tree, n_neighbors = 4, leaf size= 55'
                                        , X, y, weights='distance', metric='minkowski', p=4, algorithm='kd_tree',
                                        n_neighbors=4, leaf_size=95, stratify=stratify, train_size=train_size)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance,  metric = minkowski, p=9, algorithm = ball_tree, n_neighbors = 10, leaf size= 24'
                                        , X, y, weights='distance', metric='minkowski', p=9, algorithm='ball_tree',
                                        n_neighbors=10, leaf_size=24, stratify=stratify, train_size=train_size)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance,  metric = manhattan, p=8, algorithm = ball_tree, n_neighbors = 4, leaf size= 53'
                                        , X, y, weights='distance', metric='manhattan', p=8, algorithm='ball_tree',
                                        n_neighbors=4, leaf_size=53, stratify=stratify, train_size=train_size)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance,  metric = minkowski, p=8, algorithm = auto, n_neighbors = 8, leaf size= 45'
                                        , X, y, weights='distance', metric='minkowski', p=8, algorithm='auto',
                                        n_neighbors=8, leaf_size=45, stratify=stratify, train_size=train_size)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance,  metric = manhattan, p=6, algorithm = auto, n_neighbors = 3, leaf size= 13'
                                        , X, y, weights='distance', metric='manhattan', p=6, algorithm='auto',
                                        n_neighbors=3, leaf_size=13, stratify=stratify, train_size=train_size)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance,  metric = euclidean, p=1, algorithm = ball_tree, n_neighbors = 3, leaf size= 69'
                                        , X, y, weights='distance', metric='euclidean', p=1, algorithm='ball_tree',
                                        n_neighbors=3, leaf_size=69, stratify=stratify, train_size=train_size)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance,  metric = minkowski, p=1, algorithm = ball_tree, n_neighbors = 3, leaf size= 50'
                                        , X, y, weights='distance', metric='minkowski', p=1, algorithm='ball_tree',
                                        n_neighbors=3, leaf_size=50, stratify=stratify, train_size=train_size)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance,  metric = manhattan, p=9, algorithm = kd_tree, n_neighbors = 1, leaf size= 66'
                                        , X, y, weights='distance', metric='manhattan', p=9, algorithm='kd_tree',
                                        n_neighbors=1, leaf_size=66, stratify=stratify, train_size=train_size)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance,  metric = euclidean, p=5, algorithm = ball_tree, n_neighbors = 3, leaf size= 86'
                                        , X, y, weights='distance', metric='euclidean', p=5, algorithm='ball_tree',
                                        n_neighbors=3, leaf_size=86, stratify=stratify, train_size=train_size)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance,  metric = minkowski, p=1, algorithm = auto, n_neighbors = 6, leaf size= 28'
                                        , X, y, weights='distance', metric='minkowski', p=1, algorithm='auto',
                                        n_neighbors=6, leaf_size=28, stratify=stratify, train_size=train_size)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance,  metric = minkowski, p=5, algorithm = auto, n_neighbors = 7, leaf size= 45'
                                        , X, y, weights='distance', metric='minkowski', p=5, algorithm='auto',
                                        n_neighbors=7, leaf_size=45, stratify=stratify, train_size=train_size)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance,  metric = manhattan, p=9, algorithm = auto, n_neighbors = 4, leaf size= 44'
                                        , X, y, weights='distance', metric='manhattan', p=9, algorithm='auto',
                                        n_neighbors=4, leaf_size=44, stratify=stratify, train_size=train_size)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance,  metric = manhattan, p=2, algorithm = kd_tree, n_neighbors = 6, leaf size= 87'
                                        , X, y, weights='distance', metric='manhattan', p=2, algorithm='kd_tree',
                                        n_neighbors=6, leaf_size=87, stratify=stratify, train_size=train_size)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance,  metric = euclidean, p=6, algorithm = auto, n_neighbors = 8, leaf size= 56'
                                        , X, y, weights='distance', metric='euclidean', p=6, algorithm='auto',
                                        n_neighbors=8, leaf_size=56, stratify=stratify, train_size=train_size)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance,  metric = minkowski, p=1, algorithm = auto, n_neighbors = 9, leaf size= 54'
                                        , X, y, weights='distance', metric='minkowski', p=1, algorithm='auto',
                                        n_neighbors=9, leaf_size=54, stratify=stratify, train_size=train_size)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance,  metric = minkowski, p=1, algorithm = auto, n_neighbors = 7, leaf size= 55'
                                        , X, y, weights='distance', metric='minkowski', p=1, algorithm='auto',
                                        n_neighbors=7, leaf_size=55, stratify=stratify, train_size=train_size)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance,  metric = manhattan, p=10, algorithm = auto, n_neighbors = 6, leaf size= 85'
                                        , X, y, weights='distance', metric='manhattan', p=10, algorithm='auto',
                                        n_neighbors=6, leaf_size=85, stratify=stratify, train_size=train_size)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance,  metric = euclidean, p=4, algorithm = kd_tree, n_neighbors = 3, leaf size= 1'
                                        , X, y, weights='distance', metric='euclidean', p=4, algorithm='kd_tree',
                                        n_neighbors=3, leaf_size=1, stratify=stratify, train_size=train_size)

        # run_algorithm_KNN_configuration(metrics,
        #                                 ''
        #                                         , X, y, weights='uniform', metric='manhattan', p=7, algorithm='ball_tree',
        #                                         n_neighbors=6, leaf_size=1, stratify=stratify, train_size=train_size)
        # run_algorithm_KNN_configuration(metrics,
        #                                 ''
        #                                 , X, y, weights='distance', metric='manhattan', p=4, algorithm='kd_tree',
        #                                 n_neighbors=6, leaf_size=1, stratify=stratify, train_size=train_size)
        # run_algorithm_KNN_configuration(metrics,
        #                                 ''
        #                                 , X, y, weights='uniform', metric='manhattan', p=1, algorithm='ball_tree',
        #                                 n_neighbors=6, leaf_size=91, stratify=stratify, train_size=train_size)
        # run_algorithm_KNN_configuration(metrics,
        #                                 ''
        #                                 , X, y, weights='uniform', metric='manhattan', p=9, algorithm='ball_tree',
        #                                 n_neighbors=5, leaf_size=85, stratify=stratify, train_size=train_size)
        # run_algorithm_KNN_configuration(metrics,
        #                                 ''
        #                                 , X, y, weights='uniform', metric='manhattan', p=5, algorithm='ball_tree',
        #                                 n_neighbors=9, leaf_size=1, stratify=stratify, train_size=train_size)
        # run_algorithm_KNN_configuration(metrics,
        #                                 ''
        #                                 , X, y, weights='distance', metric='minkowski', p=1, algorithm='kd_tree',
        #                                 n_neighbors=12, leaf_size=29, stratify=stratify, train_size=train_size)
        # run_algorithm_KNN_configuration(metrics,
        #                                 ''
        #                                 , X, y, weights='distance', metric='manhattan', p=5, algorithm='kd_tree',
        #                                 n_neighbors=13, leaf_size=29, stratify=stratify, train_size=train_size)
        # run_algorithm_KNN_configuration(metrics,
        #                                 ''
        #                                 , X, y, weights='uniform', metric='manhattan', p=5, algorithm='kd_tree',
        #                                 n_neighbors=7, leaf_size=87, stratify=stratify, train_size=train_size)
        # run_algorithm_KNN_configuration(metrics,
        #                                 ''
        #                                 , X, y, weights='distance', metric='manhattan', p=8, algorithm='ball_tree',
        #                                 n_neighbors=5, leaf_size=85, stratify=stratify, train_size=train_size)
        # run_algorithm_KNN_configuration(metrics,
        #                                 ''
        #                                 , X, y, weights='uniform', metric='manhattan', p=2, algorithm='ball_tree',
        #                                 n_neighbors=7, leaf_size=49, stratify=stratify, train_size=train_size)
        # run_algorithm_KNN_configuration(metrics,
        #                                 ''
        #                                 , X, y, weights='distance', metric='manhattan', p=9, algorithm='kd_tree',
        #                                 n_neighbors=3, leaf_size=21, stratify=stratify, train_size=train_size)
        # run_algorithm_KNN_configuration(metrics,
        #                                 ''
        #                                 , X, y, weights='distance', metric='manhattan', p=7, algorithm='kd_tree',
        #                                 n_neighbors=4, leaf_size=53, stratify=stratify, train_size=train_size)
        # run_algorithm_KNN_configuration(metrics,
        #                                 ''
        #                                 , X, y, weights='distance', metric='manhattan', p=4, algorithm='kd_tree',
        #                                 n_neighbors=4, leaf_size=17, stratify=stratify, train_size=train_size)
        # run_algorithm_KNN_configuration(metrics,
        #                                 ''
        #                                 , X, y, weights='distance', metric='manhattan', p=3, algorithm='kd_tree',
        #                                 n_neighbors=11, leaf_size=25, stratify=stratify, train_size=train_size)
        # run_algorithm_KNN_configuration(metrics,
        #                                 ''
        #                                 , X, y, weights='distance', metric='manhattan', p=1, algorithm='kd_tree',
        #                                 n_neighbors=9, leaf_size=73, stratify=stratify, train_size=train_size)
        # run_algorithm_KNN_configuration(metrics,
        #                                 ''
        #                                 , X, y, weights='distance', metric='manhattan', p=8, algorithm='kd_tree',
        #                                 n_neighbors=10, leaf_size=91, stratify=stratify, train_size=train_size)
        # run_algorithm_KNN_configuration(metrics,
        #                                 ''
        #                                 , X, y, weights='distance', metric='manhattan', p=4, algorithm='kd_tree',
        #                                 n_neighbors=6, leaf_size=93, stratify=stratify, train_size=train_size)
        # run_algorithm_KNN_configuration(metrics,
        #                                 ''
        #                                 , X, y, weights='uniform', metric='manhattan', p=4, algorithm='kd_tree',
        #                                 n_neighbors=4, leaf_size=41, stratify=stratify, train_size=train_size)
        # run_algorithm_KNN_configuration(metrics,
        #                                 ''
        #                                 , X, y, weights='distance', metric='manhattan', p=6, algorithm='ball_tree',
        #                                 n_neighbors=7, leaf_size=91, stratify=stratify, train_size=train_size)
        # run_algorithm_KNN_configuration(metrics,
        #                                 ''
        #                                 , X, y, weights='distance', metric='manhattan', p=5, algorithm='ball_tree',
        #                                 n_neighbors=6, leaf_size=5, stratify=stratify, train_size=train_size)

    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.groupby(['label'], as_index=False).agg({'precision': 'mean', 'recall': 'mean',
                                                                    'f1_score':'mean', 'roc_auc': 'mean',
                                                                    'n_neighbors': 'first', 'weights': 'first',
                                                                    'algorithm': 'first',  'leaf_size': 'first',
                                                                    'p': 'first', 'metric': 'first', 'metric_params': 'first'
                                                                    })
    metrics_df = compute_average_metric(metrics_df)
    metrics_df.sort_values(by =['average_metric'], ascending=False, inplace=True)
    metrics = appendMetricsTOCSV(my_filename, metrics_df, init_metrics_for_KNN, header=True)


def run_algorithm_KNN_configuration(metrics, label, X, y,
                                    n_neighbors=5, weights='uniform', algorithm='auto',
                                    leaf_size=30, p=2, metric='minkowski', metric_params=None,
                                    stratify=False, train_size=0.8):
    X_test, X_train, y_test, y_train = split_data_in_testing_training(X, y, stratify, train_size)
    # Creating the classifier object
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
                                      n_jobs=-1, leaf_size=leaf_size, p=p, metric=metric,
                                      metric_params=metric_params)

    # Performing training
    classifier.fit(X_train, y_train)

    # Make predictions
    y_pred, y_pred_probabilities = prediction(X_test, classifier)

    # Compute metrics
    precision, recall, f1, roc_auc = cal_metrics(y_test, y_pred, y_pred_probabilities, label)
    metrics['label'].append(label)
    metrics['n_neighbors'].append(n_neighbors)
    metrics['weights'].append(weights)
    metrics['algorithm'].append(algorithm)
    metrics['leaf_size'].append(leaf_size)
    metrics['p'].append(p)
    metrics['metric'].append(metric)
    metrics['metric_params'].append(metric_params)

    metrics['precision'].append(precision)
    metrics['recall'].append(recall)
    metrics['f1_score'].append(f1)
    metrics['roc_auc'].append(roc_auc)


def init_metrics_for_KNN():
    return {'label': [], 'n_neighbors': [], 'weights': [], 'algorithm': [], 'leaf_size': [],
            'p': [], 'metric': [], 'metric_params': [],
            'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': []
            }


def run_algorithm_KNN(filename='', path = '', stratify=False, train_size=0.8,
                      normalize_data=False, scaler='min-max'):
    y, X = load_normalized_dataset(file = None, normalize = normalize_data, scaler=scaler)
    metrics = init_metrics_for_KNN()

    # full_path_filename = '/content/drive/MyDrive/code/' + filename
    # path_to_script = os.path.dirname(os.path.abspath(__file__))
    my_filename = os.path.join(path, 'results/knn', filename)

    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_KNN, header=True)
    # # Default algorithm
    # run_algorithm_KNN_configuration(metrics, 'KNN', X, y, stratify=stratify, train_size=train_size)
    #
    # run_algorithm_KNN_configuration(metrics, 'KNN, weight = distance', X, y, weights='distance',
    #                                 stratify=stratify, train_size=train_size)
    #
    # # CALIBRATING n_neighbors param
    # for n_neighbors in range(1, 10, 1):
    #     run_algorithm_KNN_configuration(metrics, 'KNN, weights = uniform, n_neighbors = ' + str(n_neighbors),
    #                                     X, y, n_neighbors=n_neighbors, weights='uniform', stratify=stratify,
    #                                     train_size=train_size)
    #     run_algorithm_KNN_configuration(metrics, 'KNN, weights = distance, n_neighbors = ' + str(n_neighbors),
    #                                     X, y, n_neighbors=n_neighbors, weights='distance', stratify=stratify,
    #                                     train_size=train_size)
    # for n_neighbors in range(10, 100, 10):
    #     run_algorithm_KNN_configuration(metrics, 'KNN, weights = uniform, n_neighbors = ' + str(n_neighbors),
    #                                     X, y, n_neighbors=n_neighbors, weights='uniform', stratify=stratify,
    #                                     train_size=train_size)
    #     run_algorithm_KNN_configuration(metrics, 'KNN, weights = distance, n_neighbors = ' + str(n_neighbors),
    #                                     X, y, n_neighbors=n_neighbors, weights='distance', stratify=stratify,
    #                                     train_size=train_size)
    #
    # # CALIBRATING algorithm param
    # run_algorithm_KNN_configuration(metrics, 'KNN, weight = distance, algorithm = ball_tree',
    #                                 X, y, weights='distance', algorithm='ball_tree', stratify=stratify,
    #                                 train_size=train_size)
    # run_algorithm_KNN_configuration(metrics, 'KNN, weight = uniform, algorithm = ball_tree',
    #                                 X, y, weights='uniform', algorithm='ball_tree', stratify=stratify,
    #                                 train_size=train_size)
    # run_algorithm_KNN_configuration(metrics, 'KNN, weight = distance, algorithm = brute', X, y, weights='distance',
    #                                 algorithm='brute', stratify=stratify, train_size=train_size)
    # run_algorithm_KNN_configuration(metrics, 'KNN, weight = uniform, algorithm = brute',
    #                                 X, y, weights='uniform', algorithm='brute', stratify=stratify,
    #                                 train_size=train_size)
    # run_algorithm_KNN_configuration(metrics, 'KNN, weight = distance, algorithm = kd_tree',
    #                                 X, y, weights='distance', algorithm='kd_tree', stratify=stratify,
    #                                 train_size=train_size)
    # run_algorithm_KNN_configuration(metrics, 'KNN, weight = uniform, algorithm = kd_tree',
    #                                 X, y, weights='uniform', algorithm='kd_tree', stratify=stratify,
    #                                 train_size=train_size)
    #
    # # CALIBRATING leaf_size param
    # for leaf_size in range(1, 20, 5):
    #     run_algorithm_KNN_configuration(metrics, 'KNN, weights = uniform, algorithm = kd_tree, leaf_size = ' +
    #                                     str(leaf_size), X, y, leaf_size=leaf_size, weights='uniform',
    #                                     algorithm='kd_tree', stratify=stratify, train_size=train_size)
    #     run_algorithm_KNN_configuration(metrics, 'KNN, weights = distance, algorithm = kd_tree, leaf_size = ' +
    #                                     str(leaf_size), X, y, leaf_size=leaf_size, weights='distance',
    #                                     algorithm='kd_tree', stratify=stratify, train_size=train_size)
    #     run_algorithm_KNN_configuration(metrics, 'KNN, weights = uniform, algorithm = ball_tree, leaf_size = ' +
    #                                     str(leaf_size), X, y, leaf_size=leaf_size, weights='uniform',
    #                                     algorithm='ball_tree', stratify=stratify, train_size=train_size)
    #     run_algorithm_KNN_configuration(metrics, 'KNN, weights = distance, algorithm = ball_tree, leaf_size = ' +
    #                                     str(leaf_size), X, y, leaf_size=leaf_size, weights='distance',
    #                                     algorithm='ball_tree', stratify=stratify, train_size=train_size)
    #
    # for leaf_size in range(20, 40, 1):
    #     run_algorithm_KNN_configuration(metrics, 'KNN, weights = uniform, algorithm = kd_tree, leaf_size = ' +
    #                                     str(leaf_size), X, y, leaf_size=leaf_size, weights='uniform',
    #                                     algorithm='kd_tree', stratify=stratify, train_size=train_size)
    #     run_algorithm_KNN_configuration(metrics, 'KNN, weights = distance, algorithm = kd_tree, leaf_size = ' +
    #                                     str(leaf_size), X, y, leaf_size=leaf_size, weights='distance',
    #                                     algorithm='kd_tree', stratify=stratify, train_size=train_size)
    #     run_algorithm_KNN_configuration(metrics, 'KNN, weights = uniform, algorithm = ball_tree, leaf_size = ' +
    #                                     str(leaf_size), X, y, leaf_size=leaf_size, weights='uniform',
    #                                     algorithm='ball_tree', stratify=stratify, train_size=train_size)
    #     run_algorithm_KNN_configuration(metrics, 'KNN, weights = distance, algorithm = ball_tree, leaf_size = ' +
    #                                     str(leaf_size), X, y, leaf_size=leaf_size, weights='distance',
    #                                     algorithm='ball_tree', stratify=stratify, train_size=train_size)
    #
    # for leaf_size in range(40, 100, 10):
    #     run_algorithm_KNN_configuration(metrics, 'KNN, weights = uniform, algorithm = kd_tree, leaf_size = ' +
    #                                     str(leaf_size), X, y, leaf_size=leaf_size, weights='uniform',
    #                                     algorithm='kd_tree', stratify=stratify, train_size=train_size)
    #     run_algorithm_KNN_configuration(metrics, 'KNN, weights = distance, algorithm = kd_tree, leaf_size = ' +
    #                                     str(leaf_size), X, y, leaf_size=leaf_size, weights='distance',
    #                                     algorithm='kd_tree', stratify=stratify, train_size=train_size)
    #     run_algorithm_KNN_configuration(metrics, 'KNN, weights = uniform, algorithm = ball_tree, leaf_size = ' +
    #                                     str(leaf_size), X, y, leaf_size=leaf_size, weights='uniform',
    #                                     algorithm='ball_tree', stratify=stratify, train_size=train_size)
    #     run_algorithm_KNN_configuration(metrics, 'KNN, weights = distance, algorithm = ball_tree, leaf_size = ' +
    #                                     str(leaf_size), X, y, leaf_size=leaf_size, weights='distance',
    #                                     algorithm='ball_tree', stratify=stratify, train_size=train_size)
    #
    # # p
    # # metric -> euclidean, manhattan, chebyshev, minkowski
    # run_algorithm_KNN_configuration(metrics, 'KNN, weights = uniform, metric = euclidian ',
    #                                 X, y, weights='uniform', metric='euclidean', stratify=stratify,
    #                                 train_size=train_size)
    # run_algorithm_KNN_configuration(metrics, 'KNN, weights = distance,  metric = euclidian ', X, y,
    #                                 weights='distance', metric='euclidean', stratify=stratify,
    #                                 train_size=train_size)
    # run_algorithm_KNN_configuration(metrics, 'KNN, weights = uniform, metric = manhattan ', X, y,
    #                                 weights='uniform', metric='manhattan', stratify=stratify,
    #                                 train_size=train_size)
    # run_algorithm_KNN_configuration(metrics, 'KNN, weights = distance,  metric = manhattan ', X, y,
    #                                 weights='distance', metric='manhattan', stratify=stratify, train_size=train_size)
    # run_algorithm_KNN_configuration(metrics, 'KNN, weights = uniform, metric = chebyshev ',
    #                                 X, y, weights='uniform', metric='chebyshev', stratify=stratify,
    #                                 train_size=train_size)
    # run_algorithm_KNN_configuration(metrics, 'KNN, weights = distance,  metric = chebyshev ', X, y,
    #                                 weights='distance', metric='chebyshev', stratify=stratify,
    #                                 train_size=train_size)
    #
    # for p in range(1, 10, 1):
    #     run_algorithm_KNN_configuration(metrics, 'KNN, weights = uniform, metric = minkowski, p=' + str(p),
    #                                     X, y, weights='uniform', metric='minkowski', p=p,
    #                                     stratify=stratify, train_size=train_size)
    #     run_algorithm_KNN_configuration(metrics, 'KNN, weights = distance,  metric = minkowski, p=' +
    #                                     str(p), X, y, weights='distance', metric='minkowski',
    #                                     p=p, stratify=stratify, train_size=train_size)

    for weights in ['distance']:
        for algorithm in ['ball_tree', 'kd_tree', 'auto']:
            for metric in ['manhattan', 'euclidean', 'minkowski']:
                for p in range(1, 11):
                    for n_neighbors in range(1, 11):
                        concatenated_leaf_size = chain(range(1, 15, 2), range(20, 35, 2), range(43, 47),
                                                       range(50, 57), range(65, 70), range(85, 92))
                        for leaf_size in concatenated_leaf_size:
                            run_algorithm_KNN_configuration(metrics,
                                                            'KNN, weights = ' + weights + ',  metric = ' + metric + ', p=' +
                                                            str(p) + ', algorithm = ' + algorithm + ', n_neighbors = ' + str(
                                                                n_neighbors)
                                                            + ', leaf size= ' + str(leaf_size), X, y, weights=weights,
                                                            metric=metric,
                                                            algorithm=algorithm, n_neighbors=n_neighbors,
                                                            leaf_size=leaf_size,
                                                            p=p, stratify=stratify, train_size=train_size)
                        metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_KNN)
