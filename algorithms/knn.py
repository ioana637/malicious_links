import os

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from utils import prediction, split_data, cal_metrics, appendMetricsTOCSV

import numpy as np
from itertools import chain


def run_top_20_KNN_configs(df, filename='', stratify=False, train_size=0.8,
                           normalize_data=True, scaler='min-max', n_rep=100):
    y = df['label'].copy()
    X = df.drop('label', axis=1).copy()
    metrics = init_metrics_for_KNN()

    # full_path_filename = '/content/drive/MyDrive/code/' + filename
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    my_filename = os.path.join(path_to_script, 'knn', filename)

    for i in range(1, n_rep):
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance,  metric = manhattan, p=7, algorithm = ball_tree, n_neighbors = 9, leaf size= 73'
                                        , X, y, weights='distance', metric='manhattan', p=7, algorithm='ball_tree',
                                        n_neighbors=9, leaf_size=73, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance,  metric = manhattan, p=1, algorithm = ball_tree, n_neighbors = 7, leaf size= 41'
                                        , X, y, weights='distance', metric='manhattan', p=1, algorithm='ball_tree',
                                        n_neighbors=7, leaf_size=41, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance,  metric = manhattan, p=5, algorithm = ball_tree, n_neighbors = 3, leaf size= 21'
                                        , X, y, weights='distance', metric='manhattan', p=5, algorithm='ball_tree',
                                        n_neighbors=3, leaf_size=21, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance,  metric = manhattan, p=6, algorithm = kd_tree, n_neighbors = 5, leaf size= 25'
                                        , X, y, weights='distance', metric='manhattan', p=6, algorithm='kd_tree',
                                        n_neighbors=5, leaf_size=25, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance,  metric = manhattan, p=9, algorithm = kd_tree, n_neighbors = 12, leaf size= 53'
                                        , X, y, weights='distance', metric='manhattan', p=9, algorithm='kd_tree',
                                        n_neighbors=12, leaf_size=53, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance,  metric = manhattan, p=6, algorithm = kd_tree, n_neighbors = 11, leaf size= 9'
                                        , X, y, weights='distance', metric='manhattan', p=6, algorithm='kd_tree',
                                        n_neighbors=11, leaf_size=9, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance,  metric = minkowski, p=1, algorithm = kd_tree, n_neighbors = 10, leaf size= 53'
                                        , X, y, weights='distance', metric='minkowski', p=1, algorithm='kd_tree',
                                        n_neighbors=10, leaf_size=53, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = uniform,  metric = manhattan, p=7, algorithm = ball_tree, n_neighbors = 7, leaf size= 37'
                                        , X, y, weights='uniform', metric='manhattan', p=7, algorithm='ball_tree',
                                        n_neighbors=7, leaf_size=37, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = uniform,  metric = manhattan, p=5, algorithm = ball_tree, n_neighbors = 14, leaf size= 13'
                                        , X, y, weights='uniform', metric='manhattan', p=5, algorithm='ball_tree',
                                        n_neighbors=14, leaf_size=13, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance,  metric = manhattan, p=6, algorithm = kd_tree, n_neighbors = 6, leaf size= 5'
                                        , X, y, weights='distance', metric='manhattan', p=6, algorithm='kd_tree',
                                        n_neighbors=6, leaf_size=5, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance,  metric = manhattan, p=5, algorithm = ball_tree, n_neighbors = 10, leaf size= 1'
                                        , X, y, weights='distance', metric='manhattan', p=5, algorithm='ball_tree',
                                        n_neighbors=10, leaf_size=1, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance,  metric = manhattan, p=7, algorithm = kd_tree, n_neighbors = 7, leaf size= 17'
                                        , X, y, weights='distance', metric='manhattan', p=7, algorithm='kd_tree',
                                        n_neighbors=7, leaf_size=17, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance,  metric = manhattan, p=6, algorithm = ball_tree, n_neighbors = 9, leaf size= 37'
                                        , X, y, weights='distance', metric='manhattan', p=6, algorithm='ball_tree',
                                        n_neighbors=9, leaf_size=37, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = uniform,  metric = manhattan, p=9, algorithm = ball_tree, n_neighbors = 5, leaf size= 45'
                                        , X, y, weights='uniform', metric='manhattan', p=9, algorithm='ball_tree',
                                        n_neighbors=5, leaf_size=45, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = uniform,  metric = manhattan, p=3, algorithm = ball_tree, n_neighbors = 8, leaf size= 69'
                                        , X, y, weights='uniform', metric='manhattan', p=3, algorithm='ball_tree',
                                        n_neighbors=8, leaf_size=69, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance,  metric = manhattan, p=1, algorithm = ball_tree, n_neighbors = 6, leaf size= 37'
                                        , X, y, weights='distance', metric='manhattan', p=1, algorithm='ball_tree',
                                        n_neighbors=6, leaf_size=37, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance,  metric = manhattan, p=7, algorithm = ball_tree, n_neighbors = 9, leaf size= 69'
                                        , X, y, weights='distance', metric='manhattan', p=7, algorithm='ball_tree',
                                        n_neighbors=9, leaf_size=69, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance,  metric = minkowski, p=2, algorithm = kd_tree, n_neighbors = 8, leaf size= 73'
                                        , X, y, weights='distance', metric='minkowski', p=2, algorithm='kd_tree',
                                        n_neighbors=8, leaf_size=73, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance,  metric = manhattan, p=1, algorithm = kd_tree, n_neighbors = 7, leaf size= 73'
                                        , X, y, weights='distance', metric='manhattan', p=1, algorithm='kd_tree',
                                        n_neighbors=7, leaf_size=73, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance,  metric = manhattan, p=6, algorithm = ball_tree, n_neighbors = 12, leaf size= 71'
                                        , X, y, weights='distance', metric='manhattan', p=6, algorithm='ball_tree',
                                        n_neighbors=12, leaf_size=71, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)

        run_algorithm_KNN_configuration(metrics,
                                                'KNN, weights = uniform,  metric = manhattan, p=7, algorithm = ball_tree, n_neighbors = 6, leaf size= 1'
                                                , X, y, weights='uniform', metric='manhattan', p=7, algorithm='ball_tree',
                                                n_neighbors=6, leaf_size=1, stratify=stratify, train_size=train_size,
                                                normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance,  metric = manhattan, p=4, algorithm = kd_tree, n_neighbors = 6, leaf size= 1'
                                        , X, y, weights='distance', metric='manhattan', p=4, algorithm='kd_tree',
                                        n_neighbors=6, leaf_size=1, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = uniform, metric = manhattan, p = 1, algorithm = ball_tree, n_neighbors = 6, leaf_size = 91'
                                        , X, y, weights='uniform', metric='manhattan', p=1, algorithm='ball_tree',
                                        n_neighbors=6, leaf_size=91, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = uniform, metric = manhattan, p = 9, algorithm = ball_tree, n_neighbors = 5, leaf_size = 85'
                                        , X, y, weights='uniform', metric='manhattan', p=9, algorithm='ball_tree',
                                        n_neighbors=5, leaf_size=85, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = uniform, metric = manhattan, p = 5, algorithm = ball_tree, n_neighbors = 9, leaf_size = 1'
                                        , X, y, weights='uniform', metric='manhattan', p=5, algorithm='ball_tree',
                                        n_neighbors=9, leaf_size=1, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance, metric = minkowski, p = 1, algorithm = kd_tree, n_neighbors = 12, leaf_size = 29'
                                        , X, y, weights='distance', metric='minkowski', p=1, algorithm='kd_tree',
                                        n_neighbors=12, leaf_size=29, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance, metric = manhattan, p = 5, algorithm = kd_tree, n_neighbors = 13, leaf_size = 29'
                                        , X, y, weights='distance', metric='manhattan', p=5, algorithm='kd_tree',
                                        n_neighbors=13, leaf_size=29, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = uniform, metric = manhattan, p = 5, algorithm = kd_tree, n_neighbors = 7, leaf_size = 87'
                                        , X, y, weights='uniform', metric='manhattan', p=5, algorithm='kd_tree',
                                        n_neighbors=7, leaf_size=87, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance, metric = manhattan, p = 8, algorithm = ball_tree, n_neighbors = 5, leaf_size = 85'
                                        , X, y, weights='distance', metric='manhattan', p=8, algorithm='ball_tree',
                                        n_neighbors=5, leaf_size=85, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = uniform, metric = manhattan, p = 2, algorithm = ball_tree, n_neighbors = 7, leaf_size = 49'
                                        , X, y, weights='uniform', metric='manhattan', p=2, algorithm='ball_tree',
                                        n_neighbors=7, leaf_size=49, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance, metric = manhattan, p = 9, algorithm = kd_tree, n_neighbors = 3, leaf_size = 21'
                                        , X, y, weights='distance', metric='manhattan', p=9, algorithm='kd_tree',
                                        n_neighbors=3, leaf_size=21, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance, metric = manhattan, p = 7, algorithm = kd_tree, n_neighbors = 4, leaf_size = 53'
                                        , X, y, weights='distance', metric='manhattan', p=7, algorithm='kd_tree',
                                        n_neighbors=4, leaf_size=53, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance, metric = manhattan, p = 4, algorithm = kd_tree, n_neighbors = 4, leaf_size = 17'
                                        , X, y, weights='distance', metric='manhattan', p=4, algorithm='kd_tree',
                                        n_neighbors=4, leaf_size=17, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance, metric = manhattan, p = 3, algorithm = kd_tree, n_neighbors = 11, leaf_size = 25'
                                        , X, y, weights='distance', metric='manhattan', p=3, algorithm='kd_tree',
                                        n_neighbors=11, leaf_size=25, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance, metric = manhattan, p = 1, algorithm = kd_tree, n_neighbors = 9, leaf_size = 73'
                                        , X, y, weights='distance', metric='manhattan', p=1, algorithm='kd_tree',
                                        n_neighbors=9, leaf_size=73, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance, metric = manhattan, p = 8, algorithm = kd_tree, n_neighbors = 10, leaf_size = 91'
                                        , X, y, weights='distance', metric='manhattan', p=8, algorithm='kd_tree',
                                        n_neighbors=10, leaf_size=91, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance, metric = manhattan, p = 4, algorithm = kd_tree, n_neighbors = 6, leaf_size = 93'
                                        , X, y, weights='distance', metric='manhattan', p=4, algorithm='kd_tree',
                                        n_neighbors=6, leaf_size=93, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = uniform, metric = manhattan, p = 4, algorithm = kd_tree, n_neighbors = 4, leaf_size = 41'
                                        , X, y, weights='uniform', metric='manhattan', p=4, algorithm='kd_tree',
                                        n_neighbors=4, leaf_size=41, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance, metric = manhattan, p = 6, algorithm = ball_tree, n_neighbors = 7, leaf_size = 91'
                                        , X, y, weights='distance', metric='manhattan', p=6, algorithm='ball_tree',
                                        n_neighbors=7, leaf_size=91, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics,
                                        'KNN, weights = distance, metric = manhattan, p = 5, algorithm = ball_tree, n_neighbors = 6, leaf_size = 5'
                                        , X, y, weights='distance', metric='manhattan', p=5, algorithm='ball_tree',
                                        n_neighbors=6, leaf_size=5, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)

    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.groupby(['label'], as_index=False).agg({'precision': 'mean', 'recall': 'mean',
                                                                    'f1_score':'mean', 'roc_auc': 'mean',
                                                                    'n_neighbors': 'first', 'weights': 'first',
                                                                    'algorithm': 'first',  'leaf_size': 'first',
                                                                    'p': 'first', 'metric': 'first', 'metric_params': 'first'
                                                                    })
    metrics = appendMetricsTOCSV(my_filename, metrics_df, init_metrics_for_KNN, header=True)


def run_algorithm_KNN_configuration(metrics, label, X, y,
                                    n_neighbors=5, weights='uniform', algorithm='auto',
                                    leaf_size=30, p=2, metric='minkowski', metric_params=None,
                                    stratify=False, train_size=0.8, normalize_data=False, scaler='min-max'):
    X_train, X_test, y_train, y_test = split_data(X, y, normalize_data=normalize_data, stratify=stratify,
                                                  train_size=train_size, scaler=scaler);

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


def run_algorithm_KNN(df, filename='', stratify=False, train_size=0.8,
                      normalize_data=False, scaler='min-max'):
    y = df['label'].copy()
    X = df.drop('label', axis=1).copy()
    metrics = init_metrics_for_KNN()

    # full_path_filename = '/content/drive/MyDrive/code/' + filename
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    my_filename = os.path.join(path_to_script, 'knn', filename)

    # Default algorithm
    run_algorithm_KNN_configuration(metrics, 'KNN', X, y, stratify=stratify, train_size=train_size,
                                    normalize_data=normalize_data, scaler=scaler)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_KNN, header=True)

    run_algorithm_KNN_configuration(metrics, 'KNN, weight = distance', X, y, weights='distance',
                                    stratify=stratify, train_size=train_size,
                                    normalize_data=normalize_data, scaler=scaler)

    # CALIBRATING n_neighbors param
    for n_neighbors in range(1, 10, 1):
        run_algorithm_KNN_configuration(metrics, 'KNN, weights = uniform, n_neighbors = ' + str(n_neighbors),
                                        X, y, n_neighbors=n_neighbors, weights='uniform', stratify=stratify,
                                        train_size=train_size, normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics, 'KNN, weights = distance, n_neighbors = ' + str(n_neighbors),
                                        X, y, n_neighbors=n_neighbors, weights='distance', stratify=stratify,
                                        train_size=train_size, normalize_data=normalize_data, scaler=scaler)
    for n_neighbors in range(10, 100, 10):
        run_algorithm_KNN_configuration(metrics, 'KNN, weights = uniform, n_neighbors = ' + str(n_neighbors),
                                        X, y, n_neighbors=n_neighbors, weights='uniform', stratify=stratify,
                                        train_size=train_size, normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics, 'KNN, weights = distance, n_neighbors = ' + str(n_neighbors),
                                        X, y, n_neighbors=n_neighbors, weights='distance', stratify=stratify,
                                        train_size=train_size, normalize_data=normalize_data, scaler=scaler)

    # CALIBRATING algorithm param
    run_algorithm_KNN_configuration(metrics, 'KNN, weight = distance, algorithm = ball_tree',
                                    X, y, weights='distance', algorithm='ball_tree', stratify=stratify,
                                    train_size=train_size, normalize_data=normalize_data, scaler=scaler)
    run_algorithm_KNN_configuration(metrics, 'KNN, weight = uniform, algorithm = ball_tree',
                                    X, y, weights='uniform', algorithm='ball_tree', stratify=stratify,
                                    train_size=train_size, normalize_data=normalize_data, scaler=scaler)
    run_algorithm_KNN_configuration(metrics, 'KNN, weight = distance, algorithm = brute', X, y, weights='distance',
                                    algorithm='brute', stratify=stratify, train_size=train_size,
                                    normalize_data=normalize_data, scaler=scaler)
    run_algorithm_KNN_configuration(metrics, 'KNN, weight = uniform, algorithm = brute',
                                    X, y, weights='uniform', algorithm='brute', stratify=stratify,
                                    train_size=train_size, normalize_data=normalize_data, scaler=scaler)
    run_algorithm_KNN_configuration(metrics, 'KNN, weight = distance, algorithm = kd_tree',
                                    X, y, weights='distance', algorithm='kd_tree', stratify=stratify,
                                    train_size=train_size, normalize_data=normalize_data, scaler=scaler)
    run_algorithm_KNN_configuration(metrics, 'KNN, weight = uniform, algorithm = kd_tree',
                                    X, y, weights='uniform', algorithm='kd_tree', stratify=stratify,
                                    train_size=train_size, normalize_data=normalize_data, scaler=scaler)

    # CALIBRATING leaf_size param
    for leaf_size in range(1, 20, 5):
        run_algorithm_KNN_configuration(metrics, 'KNN, weights = uniform, algorithm = kd_tree, leaf_size = ' +
                                        str(leaf_size), X, y, leaf_size=leaf_size, weights='uniform',
                                        algorithm='kd_tree', stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics, 'KNN, weights = distance, algorithm = kd_tree, leaf_size = ' +
                                        str(leaf_size), X, y, leaf_size=leaf_size, weights='distance',
                                        algorithm='kd_tree', stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics, 'KNN, weights = uniform, algorithm = ball_tree, leaf_size = ' +
                                        str(leaf_size), X, y, leaf_size=leaf_size, weights='uniform',
                                        algorithm='ball_tree', stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics, 'KNN, weights = distance, algorithm = ball_tree, leaf_size = ' +
                                        str(leaf_size), X, y, leaf_size=leaf_size, weights='distance',
                                        algorithm='ball_tree', stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)

    for leaf_size in range(20, 40, 1):
        run_algorithm_KNN_configuration(metrics, 'KNN, weights = uniform, algorithm = kd_tree, leaf_size = ' +
                                        str(leaf_size), X, y, leaf_size=leaf_size, weights='uniform',
                                        algorithm='kd_tree', stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics, 'KNN, weights = distance, algorithm = kd_tree, leaf_size = ' +
                                        str(leaf_size), X, y, leaf_size=leaf_size, weights='distance',
                                        algorithm='kd_tree', stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics, 'KNN, weights = uniform, algorithm = ball_tree, leaf_size = ' +
                                        str(leaf_size), X, y, leaf_size=leaf_size, weights='uniform',
                                        algorithm='ball_tree', stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics, 'KNN, weights = distance, algorithm = ball_tree, leaf_size = ' +
                                        str(leaf_size), X, y, leaf_size=leaf_size, weights='distance',
                                        algorithm='ball_tree', stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)

    for leaf_size in range(40, 100, 10):
        run_algorithm_KNN_configuration(metrics, 'KNN, weights = uniform, algorithm = kd_tree, leaf_size = ' +
                                        str(leaf_size), X, y, leaf_size=leaf_size, weights='uniform',
                                        algorithm='kd_tree', stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics, 'KNN, weights = distance, algorithm = kd_tree, leaf_size = ' +
                                        str(leaf_size), X, y, leaf_size=leaf_size, weights='distance',
                                        algorithm='kd_tree', stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics, 'KNN, weights = uniform, algorithm = ball_tree, leaf_size = ' +
                                        str(leaf_size), X, y, leaf_size=leaf_size, weights='uniform',
                                        algorithm='ball_tree', stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics, 'KNN, weights = distance, algorithm = ball_tree, leaf_size = ' +
                                        str(leaf_size), X, y, leaf_size=leaf_size, weights='distance',
                                        algorithm='ball_tree', stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)

    # p
    # metric -> euclidean, manhattan, chebyshev, minkowski
    run_algorithm_KNN_configuration(metrics, 'KNN, weights = uniform, metric = euclidian ',
                                    X, y, weights='uniform', metric='euclidean', stratify=stratify,
                                    train_size=train_size, normalize_data=normalize_data, scaler=scaler)
    run_algorithm_KNN_configuration(metrics, 'KNN, weights = distance,  metric = euclidian ', X, y,
                                    weights='distance', metric='euclidean', stratify=stratify,
                                    train_size=train_size, normalize_data=normalize_data, scaler=scaler)
    run_algorithm_KNN_configuration(metrics, 'KNN, weights = uniform, metric = manhattan ', X, y,
                                    weights='uniform', metric='manhattan', stratify=stratify,
                                    train_size=train_size, normalize_data=normalize_data, scaler=scaler)
    run_algorithm_KNN_configuration(metrics, 'KNN, weights = distance,  metric = manhattan ', X, y,
                                    weights='distance', metric='manhattan', stratify=stratify, train_size=train_size,
                                    normalize_data=normalize_data, scaler=scaler)
    run_algorithm_KNN_configuration(metrics, 'KNN, weights = uniform, metric = chebyshev ',
                                    X, y, weights='uniform', metric='chebyshev', stratify=stratify,
                                    train_size=train_size, normalize_data=normalize_data, scaler=scaler)
    run_algorithm_KNN_configuration(metrics, 'KNN, weights = distance,  metric = chebyshev ', X, y,
                                    weights='distance', metric='chebyshev', stratify=stratify,
                                    train_size=train_size, normalize_data=normalize_data, scaler=scaler)

    for p in range(1, 10, 1):
        run_algorithm_KNN_configuration(metrics, 'KNN, weights = uniform, metric = minkowski, p=' + str(p),
                                        X, y, weights='uniform', metric='minkowski', p=p,
                                        stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_KNN_configuration(metrics, 'KNN, weights = distance,  metric = minkowski, p=' +
                                        str(p), X, y, weights='distance', metric='minkowski',
                                        p=p, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)

    for weights in ['distance', 'uniform']:
        # for weights in ['uniform']:
        for algorithm in ['ball_tree', 'kd_tree']:
            for metric in ['manhattan', 'chebyshev', 'minkowski']:
                # for p in range(1, 15):
                for p in range(1, 10):
                    # for n_neighbors in range(2, 25):
                    for n_neighbors in range(2, 15):
                        concatenated_leaf_size = chain(range(1, 55, 4), range(65, 75, 2), range(85, 95, 2))
                        for leaf_size in concatenated_leaf_size:
                            run_algorithm_KNN_configuration(metrics,
                                                            'KNN, weights = ' + weights + ',  metric = ' + metric + ', p=' +
                                                            str(p) + ', algorithm = ' + algorithm + ', n_neighbors = ' + str(
                                                                n_neighbors)
                                                            + ', leaf size= ' + str(leaf_size), X, y, weights=weights,
                                                            metric=metric,
                                                            algorithm=algorithm, n_neighbors=n_neighbors,
                                                            leaf_size=leaf_size,
                                                            p=p, stratify=stratify, train_size=train_size,
                                                            normalize_data=normalize_data, scaler=scaler)
                        metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_KNN)

    # export metrics to CSV FILE
    # df_metrics = pd.DataFrame(metrics)
    # df_metrics.to_csv('/content/drive/MyDrive/code/' + filename, encoding='utf-8', index= True)
