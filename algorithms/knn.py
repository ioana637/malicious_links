import os
from multiprocessing import Manager, Pool
from random import randint

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier

from utils.data_post import compute_average_metric
from utils.data_pre import load_normalized_dataset, split_data_in_testing_training
from utils.utils import prediction, cal_metrics, appendMetricsTOCSV, convert_metrics_to_csv, \
    listener_write_to_file

from itertools import chain


def prepare_KNN_params(row):
    params = {}
    if (row['weights'] == 'None' or row['weights'] == None or str(row['weights']) == 'nan'):
        params['weights'] = None
    else:
        params['weights'] = row['weights']
    if (row['metric_params'] == 'None' or row['metric_params'] == None or str(row['metric_params']) == 'nan'):
        params['metric_params'] = None
    else:
        params['metric_params'] = row['metric_params']
    params['n_neighbors'] = int(row['n_neighbors'])
    params['algorithm'] = row['algorithm']
    params['p'] = int(row['p'])
    params['leaf_size'] = int(row['leaf_size'])
    params['metric'] = row['metric']
    params['n_jobs'] = -1
    return params


def create_KNN_classifier(row):
    params = prepare_KNN_params(row)
    classifier = KNeighborsClassifier(n_neighbors=params['n_neighbors'], weights=params['weights'],
                                      algorithm=params['algorithm'], p=params['p'],
                                      leaf_size=params['leaf_size'], metric=params['metric'],
                                      metric_params=params['metric_params'], n_jobs=1)
    return classifier


def run_best_configs_knn(df_configs, filename='', path='', stratify=True, train_size=0.8,
                         normalize_data=True, scaler='min-max', n_rep=100):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_KNN()
    my_filename = os.path.join(path, 'new_results\\knn', filename)

    for i in range(1, n_rep):
        for index, row in df_configs.iterrows():
            # print('index' + str(index))
            # print(row)
            label = create_label_for_KNN_for_row(row)
            params = prepare_KNN_params(row)
            run_algorithm_KNN_configuration(metrics, label, X, y, n_neighbors=params['n_neighbors'],
                                            weights=params['weights'], algorithm=params['algorithm'],
                                            leaf_size=params['leaf_size'], p=params['p'], metric=params['metric'],
                                            metric_params=params['metric_params'],
                                            stratify=stratify, train_size=train_size
                                            )

    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.groupby(['label'], as_index=False).agg({'precision': 'mean', 'recall': 'mean',
                                                                    'f1_score': 'mean', 'roc_auc': 'mean',
                                                                    'n_neighbors': 'first', 'weights': 'first',
                                                                    'algorithm': 'first', 'leaf_size': 'first',
                                                                    'p': 'first', 'metric': 'first',
                                                                    'metric_params': 'first'})
    metrics_df = compute_average_metric(metrics_df)
    metrics_df.sort_values(by=['average_metric'], ascending=False, inplace=True)
    metrics = appendMetricsTOCSV(my_filename, metrics_df, init_metrics_for_KNN, header=True)



def run_algorithm_KNN_configuration(metrics, label, X, y,
                                    n_neighbors=5, weights='uniform', algorithm='auto',
                                    leaf_size=30, p=2, metric='minkowski', metric_params=None,
                                    stratify=False, train_size=0.8):
    X_train, X_test, y_train, y_test = split_data_in_testing_training(X, y, stratify, train_size)
    # Creating the classifier object
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
                                      n_jobs=-1, leaf_size=leaf_size, p=p, metric=metric,
                                      metric_params=metric_params)

    # Performing training
    classifier.fit(X_train, y_train)

    # Make predictions
    y_pred, y_pred_probabilities = prediction(X_test, classifier)

    # Compute metrics
    precision, recall, f1, roc_auc = cal_metrics(y_test, y_pred, y_pred_probabilities, label, classifier)
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


def run_algorithm_KNN_configuration_with_k_fold(X, y, n_neighbors=5, weights='uniform', algorithm='auto',
                                                leaf_size=30, p=2, metric='minkowski', metric_params=None, n_splits=5,
                                                n_repeats=10):
    # Creating the classifier object
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
                                      n_jobs=-1, leaf_size=leaf_size, p=p, metric=metric,
                                      metric_params=metric_params)

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=randint(1, 1000000))
    scores = cross_validate(classifier, X, y, scoring=['precision', 'recall', 'f1', 'roc_auc'], cv=rskf, n_jobs=-1,
                            return_train_score=False)
    # report performance
    print(scores.get('test_precision').mean())
    print(scores.get('test_recall').mean())
    print(scores.get('test_f1').mean())
    print(scores.get('test_roc_auc').mean())


def init_metrics_for_KNN():
    return {'label': [], 'n_neighbors': [], 'weights': [], 'algorithm': [], 'leaf_size': [],
            'p': [], 'metric': [], 'metric_params': [],
            'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': []
            }


def create_label_for_KNN_for_row(row_knn):
    return create_label_for_KNN(row_knn['n_neighbors'], row_knn['weights'], row_knn['algorithm'],
                                row_knn['leaf_size'], row_knn['p'], row_knn['metric'],
                                row_knn['metric_params'])


def create_label_for_KNN(n_neighbors, weights, algorithm, leaf_size, p, metric, metric_params):
    return "KNN, n_neighbors=" + str(n_neighbors) + ", weights=" + str(weights) + ", algorithm=" + str(
        algorithm) + ", leaf_size=" + str(leaf_size) + ", p=" + str(p) + ", metric=" + str(
        metric) + ", metric_params=" + str(metric_params)


def run_algorithm_knn_configuration_parallel(X, y, q_metrics,
                                             n_neighbors=5, weights='uniform', algorithm='auto',
                                             leaf_size=30, p=2, metric='minkowski', metric_params=None,
                                             stratify=False, train_size=0.8
                                             ):
    X_train, X_test, y_train, y_test = split_data_in_testing_training(X, y, stratify, train_size)
    try:
        # Creating the classifier object
        classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
                                          n_jobs=None, leaf_size=leaf_size, p=p, metric=metric,
                                          metric_params=metric_params)
        # Performing training
        classifier.fit(X_train, y_train)

        # Make predictions
        y_pred, y_pred_probabilities = prediction(X_test, classifier)

        # Compute metrics
        label = create_label_for_KNN(n_neighbors, weights, algorithm, leaf_size, p, metric, metric_params)
        precision, recall, f1, roc_auc = cal_metrics(y_test, y_pred, y_pred_probabilities, label, classifier)
        string_results_for_queue = convert_metrics_to_csv(',', label, n_neighbors, weights, algorithm, leaf_size, p,
                                                          metric, metric_params,
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


def run_algorithm_KNN_parallel(filename='', path='', stratify=False, train_size=0.8,
                               normalize_data=False, scaler='min-max', no_threads=8):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_KNN()
    my_filename = os.path.join(path, 'results/knn', filename)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_KNN, header=True)

    algorithm_list = ['ball_tree', 'kd_tree', 'auto']
    metric_list = ['manhattan', 'euclidean', 'minkowski']
    p_list = list(range(1, 11))
    n_neighbors_list = list(range(1, 11))
    leaf_size_list = list(
        chain(range(1, 15), range(20, 35), range(43, 47), range(50, 57), range(65, 70), range(85, 92)))
    # 46800

    with Manager() as manager:
        q_metrics = manager.Queue()
        jobs = []

        with Pool(no_threads) as pool:
            watcher = pool.apply_async(listener_write_to_file, (q_metrics, my_filename))
            for algorithm in algorithm_list:
                for metric in metric_list:
                    for p in p_list:
                        for n_neighbors in n_neighbors_list:
                            for leaf_size in leaf_size_list:
                                job = pool.apply_async(run_algorithm_knn_configuration_parallel,
                                                       (X, y, q_metrics,
                                                        n_neighbors, 'distance', algorithm, leaf_size, p, metric, None,
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


def run_algorithm_KNN(filename='', path='', stratify=False, train_size=0.8,
                      normalize_data=False, scaler='min-max'):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
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


def run_algorithm_KNN_with_k_fold(normalize_data=False, scaler='min-max'):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)

    run_algorithm_KNN_configuration_with_k_fold(X, y, weights='distance', metric='euclidean', p=5,
                                                algorithm='ball_tree',
                                                n_neighbors=3, leaf_size=86,
                                                n_splits=2, n_repeats=10)
