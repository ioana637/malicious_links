import os
from multiprocessing import Manager, Pool
from random import randint

import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier

from data_post import compute_average_metric
from data_pre import load_normalized_dataset, split_data_in_testing_training
from utils import cal_metrics, prediction, appendMetricsTOCSV, convert_metrics_to_csv, \
    listener_write_to_file
import time
import matplotlib.pyplot as plt
from itertools import chain
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate


def prepare_DT_params(row):
    params_dict = {}
    if (row['max_depth'] == 'None' or row['max_depth'] == None or str(row['max_depth']) == 'nan'):
        params_dict['max_depth'] = None
    else:
        params_dict['max_depth'] = int(row['max_depth'])

    if (row['max_leaf_nodes'] == 'None' or row['max_leaf_nodes'] == None or str(row['max_leaf_nodes']) == 'nan'):
        params_dict['max_leaf_nodes'] = None
    else:
        params_dict['max_leaf_nodes'] = int(row['max_leaf_nodes'])

    if (row['max_features'] == 'None' or row['max_features'] == None or str(row['max_features']) == 'nan'):
        params_dict['max_features'] = None
    elif (row['max_features'] == 'auto' or row['max_features'] == 'sqrt' or row['max_features'] == 'log2'):
        params_dict['max_features'] = row['max_features']
    else:
        params_dict['max_features'] = float(row['max_features'])
    if (float(row['min_samples_split']) < 1.0):
        params_dict['min_samples_split'] = float(row['min_samples_split'])
    else:
        params_dict['min_samples_split'] = int(row['min_samples_split'])

    if (float(row['min_samples_leaf']) < 1.0):
        params_dict['min_samples_leaf'] = float(row['min_samples_leaf'])
    else:
        params_dict['min_samples_leaf'] = int(row['min_samples_leaf'])
    params_dict['criterion'] = row['criterion']
    params_dict['splitter'] = row['splitter']
    params_dict['min_weight_fraction_leaf'] = float(row['min_weight_fraction_leaf'])
    params_dict['min_impurity_decrease'] = float(row['min_impurity_decrease'])
    params_dict['ccp_alpha'] = float(row['ccp_alpha'])
    params_dict['class_weight'] = row['class_weight']
    return params_dict


def create_DT_classifier(row):
    params = prepare_DT_params(row)
    classifier = DecisionTreeClassifier(criterion=params['criterion'], max_depth=params['max_depth'],
                                        min_samples_leaf=params['min_samples_leaf'],
                                        splitter=params['splitter'],
                                        min_samples_split=params['min_samples_split'],
                                        min_weight_fraction_leaf=params['min_weight_fraction_leaf'],
                                        max_features=params['max_features'],
                                        max_leaf_nodes=params['max_leaf_nodes'],
                                        min_impurity_decrease=params['min_impurity_decrease'],
                                        ccp_alpha=params['ccp_alpha'], class_weight=params['class_weight'])
    return classifier


def run_best_configs_DT(df_configs, filename='', path='', stratify=True, train_size=0.8,
                        normalize_data=True, scaler='min-max', n_rep=100):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_DT()
    my_filename = os.path.join(path, 'new_results\\dt', filename)

    for i in range(1, n_rep):
        for index, row in df_configs.iterrows():
            # print('index' + str(index))
            # print(row)
            label = create_label_for_DT_for_row(row)
            params = prepare_DT_params(row)
            run_algorithm_dt_configuration(metrics, label, X, y,
                                           criterion=params['criterion'],
                                           splitter=params['splitter'], max_depth=params['max_depth'],
                                           min_samples_leaf=params['min_samples_leaf'],
                                           min_samples_split=params['min_samples_split'],
                                           min_weight_fraction_leaf=params['min_weight_fraction_leaf'],
                                           max_features=params['max_features'], max_leaf_nodes=params['max_leaf_nodes'],
                                           min_impurity_decrease=params['min_impurity_decrease'],
                                           class_weight=params['class_weight'], ccp_alpha=params['ccp_alpha'],
                                           stratify=stratify, train_size=train_size
                                           )

    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.groupby(['label'], as_index=False).agg({'precision': 'mean', 'recall': 'mean',
                                                                    'f1_score': 'mean', 'roc_auc': 'mean',
                                                                    'criterion': 'first', 'splitter': 'first',
                                                                    'max_depth': 'first', 'min_samples_leaf': 'first',
                                                                    'min_samples_split': 'first',
                                                                    'min_weight_fraction_leaf': 'first',
                                                                    'max_features': 'first',
                                                                    'max_leaf_nodes': 'first',
                                                                    'min_impurity_decrease': 'first',
                                                                    'class_weight': 'first',
                                                                    'ccp_alpha': 'first'
                                                                    })
    metrics_df = compute_average_metric(metrics_df)
    metrics_df.sort_values(by=['average_metric'], ascending=False, inplace=True)
    metrics = appendMetricsTOCSV(my_filename, metrics_df, init_metrics_for_DT, header=True)


def run_algorithm_dt_configuration_feature_importance(criterion='gini',
                                                      splitter='best', max_depth=12, min_samples_leaf=32,
                                                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                                                      max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
                                                      class_weight=None, ccp_alpha=0.0,
                                                      train_size=0.8, stratify=False,
                                                      normalize_data=False, scaler='min-max'
                                                      ):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    feature_names = X.columns
    print(feature_names)
    X_train, X_test, y_train, y_test = split_data_in_testing_training(X, y, stratify, train_size)

    # Creating the classifier object
    classifier = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                        splitter=splitter, min_samples_split=min_samples_split,
                                        min_weight_fraction_leaf=min_weight_fraction_leaf,
                                        max_features=max_features, max_leaf_nodes=max_leaf_nodes,
                                        min_impurity_decrease=min_impurity_decrease,
                                        class_weight=class_weight, ccp_alpha=ccp_alpha,
                                        )

    # Performing training
    classifier.fit(X_train, y_train)

    start_time = time.time()
    importances = classifier.feature_importances_
    std = np.std(importances, axis=0)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
    tree_importances = pd.Series(importances, index=feature_names)
    fig, ax = plt.subplots()
    tree_importances.plot.bar(yerr=std, ax=ax)
    # ax.set_title("Feature importances using MDI (Decision Tree)")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

    #     ---------------------------------------------------------
    start_time = time.time()
    result = permutation_importance(
        classifier, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
    )
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    decision_tree_importances = pd.Series(result.importances_mean, index=feature_names)
    fig, ax = plt.subplots()
    decision_tree_importances.plot.bar(yerr=result.importances_std, ax=ax)
    # ax.set_title("Feature importances using permutation on full model\n(Decision Tree)")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()


def create_label_for_DT_for_row(row_dt):
    return create_label_for_DT(row_dt['criterion'], row_dt['splitter'], row_dt['max_depth'],
                               row_dt['min_samples_leaf'], row_dt['min_samples_split'],
                               row_dt['min_weight_fraction_leaf'],
                               row_dt['max_features'], row_dt['max_leaf_nodes'], row_dt['min_impurity_decrease'],
                               'balanced',
                               row_dt['ccp_alpha'])


def create_label_for_DT(criterion, splitter, max_depth, min_samples_leaf, min_samples_split, min_weight_fraction_leaf,
                        max_features, max_leaf_nodes, min_impurity_decrease, class_weight, ccp_alpha):
    return "DT, criterion=" + str(criterion) + ", splitter=" + str(splitter) + ", max_depth=" + str(
        max_depth) + ", min_samples_leaf=" + str(min_samples_leaf) + ", min_samples_split=" + str(
        min_samples_split) + ", min_weight_fraction_leaf=" + str(min_weight_fraction_leaf) + ", max_features=" + str(
        max_features) + ", max_leaf_nodes=" + str(max_leaf_nodes) + ", min_impurity_decrease=" + str(
        min_impurity_decrease) + ", class_weight=" + str(class_weight) + ", ccp_alpha=" + str(ccp_alpha)


def run_algorithm_dt_configuration_parallel(X, y, q_metrics,
                                            criterion='gini', splitter='best', max_depth=12, min_samples_leaf=32,
                                            min_samples_split=2, min_weight_fraction_leaf=0.0, max_features=None,
                                            max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight='balanced',
                                            ccp_alpha=0.0, stratify=False, train_size=0.8):
    X_train, X_test, y_train, y_test = split_data_in_testing_training(X, y, stratify, train_size)
    try:
        # Creating the classifier object
        classifier = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                            splitter=splitter, min_samples_split=min_samples_split,
                                            min_weight_fraction_leaf=min_weight_fraction_leaf,
                                            max_features=max_features, max_leaf_nodes=max_leaf_nodes,
                                            min_impurity_decrease=min_impurity_decrease,
                                            class_weight=class_weight, ccp_alpha=ccp_alpha
                                            )
        # Performing training
        classifier.fit(X_train, y_train)

        # Make predictions
        y_pred, y_pred_probabilities = prediction(X_test, classifier)

        # Compute metrics
        label = create_label_for_DT(criterion='gini', splitter='best', max_depth=12, min_samples_leaf=32,
                                    min_samples_split=2, min_weight_fraction_leaf=0.0, max_features=None,
                                    max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight='balanced',
                                    ccp_alpha=0.0)
        precision, recall, f1, roc_auc = cal_metrics(y_test, y_pred, y_pred_probabilities, label, classifier)
        string_results_for_queue = convert_metrics_to_csv(',', label,
                                                          criterion, splitter, max_depth, min_samples_leaf,
                                                          min_samples_split, min_weight_fraction_leaf, max_features,
                                                          max_leaf_nodes, min_impurity_decrease, class_weight,
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


def run_algorithm_dt_configuration(metrics, label, X, y, criterion='gini',
                                   splitter='best', max_depth=12, min_samples_leaf=32,
                                   min_samples_split=2, min_weight_fraction_leaf=0.0,
                                   max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
                                   class_weight='balanced', ccp_alpha=0.0,
                                   train_size=0.8, stratify=False
                                   ):
    X_train, X_test, y_train, y_test = split_data_in_testing_training(X, y, stratify, train_size)

    # Creating the classifier object
    classifier = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                        splitter=splitter, min_samples_split=min_samples_split,
                                        min_weight_fraction_leaf=min_weight_fraction_leaf,
                                        max_features=max_features, max_leaf_nodes=max_leaf_nodes,
                                        min_impurity_decrease=min_impurity_decrease,
                                        class_weight=class_weight, ccp_alpha=ccp_alpha
                                        )

    # Performing training
    classifier.fit(X_train, y_train)

    # Make predictions
    y_pred, y_pred_probabilities = prediction(X_test, classifier)

    # Compute metrics
    precision, recall, f1, roc_auc = cal_metrics(y_test, y_pred, y_pred_probabilities, label, classifier)
    metrics['label'].append(label)

    metrics['criterion'].append(criterion)
    metrics['splitter'].append(splitter)
    metrics['max_depth'].append(max_depth)
    metrics['min_samples_leaf'].append(min_samples_leaf)
    metrics['min_samples_split'].append(min_samples_split)
    metrics['min_weight_fraction_leaf'].append(min_weight_fraction_leaf)
    metrics['max_features'].append(max_features)
    metrics['max_leaf_nodes'].append(max_leaf_nodes)
    metrics['min_impurity_decrease'].append(min_impurity_decrease)
    metrics['class_weight'].append(class_weight)
    metrics['ccp_alpha'].append(ccp_alpha)

    metrics['precision'].append(precision)
    metrics['recall'].append(recall)
    metrics['f1_score'].append(f1)
    metrics['roc_auc'].append(roc_auc)


def init_metrics_for_DT():
    return {'label': [], 'criterion': [], 'splitter': [], 'max_depth': [],
            'min_samples_leaf': [], 'min_samples_split': [], 'min_weight_fraction_leaf': [],
            'max_features': [], 'max_leaf_nodes': [], 'min_impurity_decrease': [], 'class_weight': [],
            'ccp_alpha': [],
            'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': []
            }


def run_algorithm_dt_parallel(filename='', path='', stratify=False, train_size=0.8,
                              normalize_data=False, scaler='min-max', no_threads=8):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_DT()
    my_filename = os.path.join(path, 'results/dt', filename)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_DT, header=True)

    criterion_list = ['gini', 'entropy']
    max_depth_list = list(range(5, 16))
    min_samples_leaf_list = list(chain(range(1, 21), range(30, 36)))
    min_samples_split_list = list(range(2, 11))
    max_features_list = [None, 'sqrt']
    max_leaf_nodes_list = list(range(18, 26))
    # 82368

    with Manager() as manager:
        q_metrics = manager.Queue()
        jobs = []

        with Pool(no_threads) as pool:
            watcher = pool.apply_async(listener_write_to_file, (q_metrics, my_filename))
            for criterion in criterion_list:
                for max_depth in max_depth_list:
                    for min_samples_leaf in min_samples_leaf_list:
                        for min_samples_split in min_samples_split_list:
                            for max_features in max_features_list:
                                for max_leaf_nodes in max_leaf_nodes_list:
                                    job = pool.apply_async(run_algorithm_dt_configuration_parallel,
                                                           (X, y, q_metrics,
                                                            criterion, 'best', max_depth, min_samples_leaf,
                                                            min_samples_split, 0.0, max_features, max_leaf_nodes, 0.0,
                                                            'balanced', 0.0,
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


def run_algorithm_dt(filename='', path='', stratify=False, train_size=0.8,
                     normalize_data=False, scaler='min-max'):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_DT()

    # path_to_script = os.path.dirname(os.path.abspath(__file__))
    my_filename = os.path.join(path, 'results/dt', filename)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_DT, header=True)

    # GINI - Equivalent for islam2019mapreduce
    # run_algorithm_dt_configuration(metrics, 'DT, GINI, min_samples_leaf=32, max_depth = 12', X, y, criterion='gini',
    #                                class_weight='balanced', stratify=stratify, train_size=train_size)
    # # ENTROPY with the rest configs as for islam2019mapreduce
    # run_algorithm_dt_configuration(metrics, 'DT, ENTROPY, min_samples_leaf=32, max_depth = 12', X, y,
    #                                class_weight='balanced', criterion='entropy', stratify=stratify, train_size=train_size)
    #
    # # CALIBRATING min_samples_leaf param + ENTROPY/GINI
    # for min_samples_leaf in range(1, 100):
    #     run_algorithm_dt_configuration(metrics,
    #                                    'DT, ENTROPY, min_samples_leaf=' + str(min_samples_leaf) + ', max_depth = 12', X,
    #                                    y, criterion='entropy', min_samples_leaf=min_samples_leaf, stratify=stratify,
    #                                    class_weight='balanced', train_size=train_size)
    #     run_algorithm_dt_configuration(metrics,
    #                                    'DT, GINI, min_samples_leaf=' + str(min_samples_leaf) + ', max_depth = 12', X, y,
    #                                    criterion='gini', min_samples_leaf=min_samples_leaf, stratify=stratify,
    #                                    class_weight='balanced', train_size=train_size)
    #
    # # CALIBRATING max_depth param + GINI/ENTROPY
    # for max_depth in range(1, 100):
    #     run_algorithm_dt_configuration(metrics, 'DT, ENTROPY, min_samples_leaf= 32, max_depth = ' + str(max_depth), X,
    #                                    y, criterion='entropy', max_depth=max_depth, class_weight='balanced', stratify=stratify,
    #                                    train_size=train_size)
    #     run_algorithm_dt_configuration(metrics, 'DT, GINI, min_samples_leaf=32, max_depth = ' + str(max_depth), X, y,
    #                                    criterion='gini', max_depth=max_depth, class_weight='balanced', stratify=stratify, train_size=train_size)
    #
    # # CALIBRATING min_samples_split param + ENTROPY/GINI
    # for min_samples_split in range(2, len(X), 20):
    #     run_algorithm_dt_configuration(metrics,
    #                                    'DT, ENTROPY, min_samples_leaf= 32, max_depth = 12, min_samples_split = ' + str(
    #                                        min_samples_split), X, y, criterion='entropy',
    #                                    min_samples_split=min_samples_split,class_weight='balanced', stratify=stratify, train_size=train_size)
    #     run_algorithm_dt_configuration(metrics,
    #                                    'DT, GINI, min_samples_leaf=32, max_depth =12, min_samples_split = ' + str(
    #                                        min_samples_split), X, y, criterion='gini', class_weight='balanced',
    #                                    min_samples_split=min_samples_split, stratify=stratify, train_size=train_size)
    # metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_DT)
    # # CALIBRATING min_weight_fraction_leaf param + ENTROPY/GINI
    # for min_weight_fraction_leaf in [0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5]:
    #     run_algorithm_dt_configuration(metrics,
    #                                    'DT, ENTROPY, min_samples_leaf= 32, max_depth = 12, min_weight_fraction_leaf = ' + str(
    #                                        min_weight_fraction_leaf), X, y, criterion='entropy',
    #                                    min_weight_fraction_leaf=min_weight_fraction_leaf,class_weight='balanced', stratify=stratify,
    #                                    train_size=train_size)
    #     run_algorithm_dt_configuration(metrics,
    #                                    'DT, GINI, min_samples_leaf=32, max_depth =12, min_weight_fraction_leaf = ' + str(
    #                                        min_weight_fraction_leaf), X, y, criterion='gini', class_weight='balanced',
    #                                    min_weight_fraction_leaf=min_weight_fraction_leaf, stratify=stratify,
    #                                    train_size=train_size)
    #
    # # CALIBRATING max_features param + ENTROPY/GINI
    # for max_features in range(1, 20):
    #     run_algorithm_dt_configuration(metrics,
    #                                    'DT, ENTROPY, min_samples_leaf= 32, max_depth = 12, max_features = ' + str(
    #                                        max_features), X, y, criterion='entropy', max_features=max_features,
    #                                    class_weight='balanced',
    #                                    stratify=stratify, train_size=train_size)
    #     run_algorithm_dt_configuration(metrics, 'DT, GINI, min_samples_leaf=32, max_depth =12, max_features = ' + str(
    #         max_features), X, y, criterion='gini', max_features=max_features,
    #                                    class_weight='balanced', stratify=stratify, train_size=train_size)
    # run_algorithm_dt_configuration(metrics, 'DT, ENTROPY, min_samples_leaf= 32, max_depth = 12, max_features = log2', X,
    #                                y, criterion='entropy', max_features='log2',
    #                                class_weight='balanced', stratify=stratify,
    #                                train_size=train_size)
    # run_algorithm_dt_configuration(metrics, 'DT, GINI, min_samples_leaf=32, max_depth =12, max_features = log2', X, y,
    #                                criterion='gini', max_features='log2',
    #                                class_weight='balanced', stratify=stratify, train_size=train_size)
    # run_algorithm_dt_configuration(metrics, 'DT, ENTROPY, min_samples_leaf= 32, max_depth = 12, max_features = auto', X,
    #                                y, criterion='entropy', max_features='auto', stratify=stratify,
    #                                class_weight='balanced', train_size=train_size)
    # run_algorithm_dt_configuration(metrics, 'DT, GINI, min_samples_leaf=32, max_depth =12, max_features = auto', X, y,
    #                                class_weight='balanced', criterion='gini', max_features='auto', stratify=stratify, train_size=train_size)
    # run_algorithm_dt_configuration(metrics, 'DT, ENTROPY, min_samples_leaf= 32, max_depth = 12, max_features = sqrt', X,
    #                                y, criterion='entropy', max_features='sqrt', stratify=stratify,
    #                                class_weight='balanced', train_size=train_size)
    # run_algorithm_dt_configuration(metrics, 'DT, GINI, min_samples_leaf=32, max_depth =12, max_features = sqrt', X, y,
    #                                criterion='gini', max_features='sqrt',class_weight='balanced', stratify=stratify, train_size=train_size)
    #
    # # CALIBRATING max_leaf_nodes param + ENTROPY/GINI
    # for max_leaf_nodes in range(2, 100, 2):
    #     run_algorithm_dt_configuration(metrics,
    #                                    'DT, ENTROPY, min_samples_leaf= 32, max_depth = 12, max_leaf_nodes = ' + str(
    #                                        max_leaf_nodes), X, y, criterion='entropy', max_leaf_nodes=max_leaf_nodes,
    #                                    class_weight='balanced',stratify=stratify, train_size=train_size)
    #     run_algorithm_dt_configuration(metrics, 'DT, GINI, min_samples_leaf=32, max_depth =12, max_leaf_nodes = ' + str(
    #         max_leaf_nodes), X, y, criterion='gini', max_leaf_nodes=max_leaf_nodes, stratify=stratify,
    #                                    class_weight='balanced',train_size=train_size)
    # metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_DT)

    # # CALIBRATING min_impurity_decrease param + ENTROPY/GINI
    # for min_impurity_decrease in range(0, len(X), 20):
    #     run_algorithm_dt_configuration(metrics,
    #                                    'DT, ENTROPY, min_samples_leaf= 32, max_depth = 12, min_impurity_decrease = ' + str(
    #                                        min_impurity_decrease), X, y, criterion='entropy',
    #                                    min_impurity_decrease=min_impurity_decrease, stratify=stratify,
    #                                    class_weight='balanced', train_size=train_size)
    #     run_algorithm_dt_configuration(metrics,
    #                                    'DT, GINI, min_samples_leaf=32, max_depth =12, min_impurity_decrease = ' + str(
    #                                        min_impurity_decrease), X, y, criterion='gini',
    #                                    min_impurity_decrease=min_impurity_decrease, stratify=stratify,
    #                                    class_weight='balanced', train_size=train_size)
    #
    # # CALIBRATING class_weight param + ENTROPY/GINI
    # run_algorithm_dt_configuration(metrics,
    #                                'DT, ENTROPY, min_samples_leaf= 32, max_depth = 12, class_weight = balanced', X, y,
    #                                criterion='entropy', class_weight='balanced', stratify=stratify,
    #                                train_size=train_size)
    # run_algorithm_dt_configuration(metrics, 'DT, GINI, min_samples_leaf=32, max_depth =12, class_weight = balanced', X,
    #                                y, criterion='gini', class_weight='balanced', stratify=stratify,
    #                                train_size=train_size)
    #
    # # CALIBRATING ccp_alpha param + ENTROPY/GINI
    # for ccp_alpha in [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1]:
    #     run_algorithm_dt_configuration(metrics, 'DT, ENTROPY, min_samples_leaf= 32, max_depth = 12, ccp_alpha = ' + str(
    #         ccp_alpha), X, y, criterion='entropy', ccp_alpha=ccp_alpha, class_weight='balanced', stratify=stratify, train_size=train_size)
    #     run_algorithm_dt_configuration(metrics,
    #                                    'DT, GINI, min_samples_leaf=32, max_depth =12, ccp_alpha = ' + str(ccp_alpha), X,
    #                                    y, criterion='gini', class_weight='balanced', ccp_alpha=ccp_alpha, stratify=stratify,
    #                                    train_size=train_size)
    # metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_DT)

    for criterion in ['gini', 'entropy']:
        for max_depth in range(5, 16):
            concatenated_min_samples_leaf = chain(range(1, 21), range(30, 36))
            for min_samples_leaf in concatenated_min_samples_leaf:
                for min_samples_split in range(2, 11):
                    for max_features in [None, 'sqrt']:
                        for max_leaf_nodes in range(18, 26, 2):
                            run_algorithm_dt_configuration(metrics, 'DT, ' + criterion + ', min_samples_leaf= '
                                                           + str(min_samples_leaf) + ', max_depth = ' + str(max_depth) +
                                                           ', max_leaf_nodes = ' + str(
                                max_leaf_nodes) + ', min_samples_split = ' +
                                                           str(min_samples_split) + ', class_weight = balanced, max_features = ' + str(
                                max_features), X, y,
                                                           criterion=criterion,
                                                           max_features=max_features,
                                                           min_samples_split=min_samples_split,
                                                           max_leaf_nodes=max_leaf_nodes, max_depth=max_depth,
                                                           min_samples_leaf=min_samples_leaf, class_weight='balanced',
                                                           stratify=stratify,
                                                           train_size=train_size)
                    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_DT)


# TODO: our proposed DT more on it
# concatenated_min_samples_leaf = chain(range(10, 15), range(30, 35))
# #  10
# for min_samples_leaf in concatenated_min_samples_leaf:
#     # 5 + 4 + 5 = 14
#     concatenated_max_depth = chain(range(10, 15), range(20, 40, 5), range(85, 95, 2))
#     for max_depth in concatenated_max_depth:
#         # 7
#         for max_leaf_nodes in range(20, 90, 10):
#             concatenated_min_samples_split = chain(range(2, 10, 2), range(10, 100, 10), range(300, 320, 5))
#             # 4+9+4=17
#             for min_samples_split in concatenated_min_samples_split:
#                 # 2.,
#                 run_algorithm_dt_configuration(metrics, 'DT, ENTROPY, min_samples_leaf= '
#                                                + str(min_samples_leaf) + ', max_depth = ' + str(max_depth) +
#                                                ', max_leaf_nodes = ' + str(max_leaf_nodes) + ', min_samples_split = ' +
#                                                str(min_samples_split) + ', class_weight = balanced', X, y,
#                                                criterion='entropy',
#                                                min_samples_split=min_samples_split,
#                                                max_leaf_nodes=max_leaf_nodes, max_depth=max_depth,
#                                                min_samples_leaf=min_samples_leaf, class_weight='balanced',
#                                                stratify=stratify,
#                                                train_size=train_size)
#                 run_algorithm_dt_configuration(metrics, 'DT, GINI, min_samples_leaf='
#                                                + str(min_samples_leaf) + ', max_depth = ' + str(max_depth) +
#                                                ', max_leaf_nodes = ' + str(max_leaf_nodes) + ', min_samples_split=' +
#                                                str(min_samples_split) + ', class_weight = balanced', X, y,
#                                                criterion='gini',
#                                                min_samples_split=min_samples_split,
#                                                max_leaf_nodes=max_leaf_nodes, max_depth=max_depth,
#                                                min_samples_leaf=min_samples_leaf, class_weight='balanced',
#                                                stratify=stratify,
#                                                train_size=train_size)
#             metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_DT)


def run_algorithm_dt_configuration_with_k_fold(X, y, criterion='gini',
                                               splitter='best', max_depth=12, min_samples_leaf=32,
                                               min_samples_split=2, min_weight_fraction_leaf=0.0,
                                               max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
                                               class_weight='balanced', ccp_alpha=0.0, n_splits=5, n_repeats=10
                                               ):
    # Creating the classifier object
    classifier = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                        splitter=splitter, min_samples_split=min_samples_split,
                                        min_weight_fraction_leaf=min_weight_fraction_leaf,
                                        max_features=max_features, max_leaf_nodes=max_leaf_nodes,
                                        min_impurity_decrease=min_impurity_decrease,
                                        class_weight=class_weight, ccp_alpha=ccp_alpha
                                        )

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=randint(1, 1000000))
    scores = cross_validate(classifier, X, y, scoring=['precision', 'recall', 'f1', 'roc_auc'], cv=rskf, n_jobs=-1,
                            return_train_score=False)
    # report performance
    print(scores.get('test_precision').mean())
    print(scores.get('test_recall').mean())
    print(scores.get('test_f1').mean())
    print(scores.get('test_roc_auc').mean())


def run_algorithm_dt_with_k_fold(filename='', path='', stratify=False, train_size=0.8,
                                 normalize_data=False, scaler='min-max'):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_DT()

    my_filename = os.path.join(path, 'results/dt', filename)
    # metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_DT, header=True)

    # "DT, entropy, min_samples_leaf= 1, max_depth = 8, max_leaf_nodes = 20, min_samples_split = 7,
    # class_weight = balanced, max_features = None"
    # BEST DT obtained
    run_algorithm_dt_configuration_with_k_fold(X, y, criterion='entropy', min_samples_leaf=1, max_depth=8,
                                               max_leaf_nodes=20, min_samples_split=7, max_features=None,
                                               class_weight='balanced', n_splits=2, n_repeats=10)
