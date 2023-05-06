import os
from multiprocessing import Manager, Pool
from random import randint

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from itertools import chain
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from data_post import compute_average_metric
from data_pre import load_normalized_dataset, split_data_in_testing_training
from utils import split_data, prediction, cal_metrics, appendMetricsTOCSV, listener_write_to_file, \
    convert_metrics_to_csv

# TODO reuse for best configs
def prepare_RF_params(row):
    params_dict = {}
    if (row['max_samples'] == 'None' or row['max_samples'] == None or str(row['max_samples']) == 'nan'):
        params_dict['max_samples'] = None
    else:
        params_dict['max_samples'] = float(row['max_samples'])

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

    params_dict['n_estimators'] = int(row['n_estimators'])
    params_dict['criterion'] = row['criterion']
    params_dict['min_samples_leaf'] = float(row['min_samples_leaf'])
    params_dict['min_samples_split'] = float(row['min_samples_split'])
    params_dict['min_weight_fraction_leaf'] = float(row['min_weight_fraction_leaf'])
    params_dict['min_impurity_decrease'] = float(row['min_impurity_decrease'])
    params_dict['bootstrap'] = bool(row['bootstrap'])
    params_dict['oob_score'] = bool(row['oob_score'])
    params_dict['ccp_alpha'] = float(row['ccp_alpha'])
    params_dict['class_weight'] = 'balanced'
    params_dict['n_jobs'] = -1
    return params_dict


def create_RF_classifier(row):
    params = prepare_RF_params(row)
    return RandomForestClassifier(n_estimators=params['n_estimators'], criterion=params['criterion'],
                                  min_samples_leaf=params['min_samples_leaf'], max_depth=params['max_depth'],
                                  min_samples_split=params['min_samples_split'], max_features=params['max_features'],
                                  max_leaf_nodes=params['max_leaf_nodes'], n_jobs=params['n_jobs'],
                                  min_weight_fraction_leaf=params['min_weight_fraction_leaf'],
                                  min_impurity_decrease=params['min_impurity_decrease'],
                                  bootstrap=params['bootstrap'], oob_score=params['oob_score'],
                                  max_samples=params['max_samples'], ccp_alpha=params['ccp_alpha'],
                                  class_weight=params['class_weight'])

def run_top_20_RFC_configs(filename='', path='', stratify=False, train_size=0.8,
                           normalize_data=True, scaler='min-max', n_rep=100):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_rfc()

    # full_path_filename = '/content/drive/MyDrive/code/' + filename
    # path_to_script = os.path.dirname(os.path.abspath(__file__))
    my_filename = os.path.join(path, 'new_results/rfc', filename)

    for i in range(1, n_rep):
        run_algorithm_rfc_configuration(metrics,
                                        'RF, entropy, n_estimators = 110, max_depth=13, min_samples_leaf=1, max_leaf_nodes=1560, min_samples_split=6, class_weight=balanced',
                                        X, y, criterion='entropy', n_estimators=110, max_depth=13, min_samples_leaf=1,
                                        max_leaf_nodes=1560, min_samples_split=6, class_weight='balanced',
                                        stratify=stratify, train_size=train_size)
        run_algorithm_rfc_configuration(metrics,
                                        'RF, entropy, n_estimators = 90, max_depth=52, min_samples_leaf=3, max_leaf_nodes=684, min_samples_split=4, class_weight=balanced ',
                                        X, y, criterion='entropy', n_estimators=90, max_depth=52, min_samples_leaf=3,
                                        max_leaf_nodes=684, min_samples_split=4, class_weight='balanced',
                                        stratify=stratify, train_size=train_size)
        run_algorithm_rfc_configuration(metrics,
                                        'RF, entropy, n_estimators = 51, max_depth=54, min_samples_leaf=1, max_leaf_nodes=1560, min_samples_split=4, class_weight=balanced',
                                        X, y, criterion='entropy', n_estimators=51, max_depth=54, min_samples_leaf=1,
                                        max_leaf_nodes=1560, min_samples_split=4, class_weight='balanced',
                                        stratify=stratify, train_size=train_size)
        run_algorithm_rfc_configuration(metrics,
                                        'RF, entropy, n_estimators = 100, max_depth=11, min_samples_leaf=1, max_leaf_nodes=1563, min_samples_split=6, class_weight=balanced',
                                        X, y, criterion='entropy', n_estimators=100, max_depth=11, min_samples_leaf=1,
                                        max_leaf_nodes=1563, min_samples_split=6, class_weight='balanced',
                                        stratify=stratify, train_size=train_size)
        run_algorithm_rfc_configuration(metrics,
                                        'RF, gini, n_estimators = 110, max_depth=12, min_samples_leaf=3, max_leaf_nodes=680, min_samples_split=8, class_weight=balanced  ',
                                        X, y, criterion='gini', n_estimators=110, max_depth=12, min_samples_leaf=3,
                                        max_leaf_nodes=680, min_samples_split=8, class_weight='balanced',
                                        stratify=stratify, train_size=train_size)
        run_algorithm_rfc_configuration(metrics,
                                        'RF, entropy, n_estimators = 90, max_depth=50, min_samples_leaf=3, max_leaf_nodes=683, min_samples_split=10, class_weight=balanced ',
                                        X, y, criterion='entropy', n_estimators=90, max_depth=50, min_samples_leaf=3,
                                        max_leaf_nodes=683, min_samples_split=10, class_weight='balanced',
                                        stratify=stratify, train_size=train_size)
        run_algorithm_rfc_configuration(metrics,
                                        'RF, entropy, n_estimators = 51, max_depth=13, min_samples_leaf=1, max_leaf_nodes=1561, min_samples_split=8, class_weight=balanced ',
                                        X, y, criterion='entropy', n_estimators=51, max_depth=13, min_samples_leaf=1,
                                        max_leaf_nodes=1561, min_samples_split=8, class_weight='balanced',
                                        stratify=stratify, train_size=train_size)
        run_algorithm_rfc_configuration(metrics,
                                        'RF, entropy, n_estimators = 90, max_depth=13, min_samples_leaf=3, max_leaf_nodes=680, min_samples_split=10, class_weight=balanced',
                                        X, y, criterion='entropy', n_estimators=90, max_depth=13, min_samples_leaf=3,
                                        max_leaf_nodes=680, min_samples_split=10, class_weight='balanced',
                                        stratify=stratify, train_size=train_size)
        run_algorithm_rfc_configuration(metrics,
                                        'RF, gini, n_estimators = 110, max_depth=14, min_samples_leaf=3, max_leaf_nodes=None, min_samples_split=4, class_weight=balanced  ',
                                        X, y, criterion='gini', n_estimators=110, max_depth=14, min_samples_leaf=3,
                                        max_leaf_nodes=None, min_samples_split=4, class_weight='balanced',
                                        stratify=stratify, train_size=train_size)
        run_algorithm_rfc_configuration(metrics,
                                        'RF, entropy, n_estimators = 50, max_depth=14, min_samples_leaf=1, max_leaf_nodes=683, min_samples_split=8, class_weight=balanced   ',
                                        X, y, criterion='entropy', n_estimators=50, max_depth=14, min_samples_leaf=1,
                                        max_leaf_nodes=683, min_samples_split=8, class_weight='balanced',
                                        stratify=stratify, train_size=train_size)
        run_algorithm_rfc_configuration(metrics,
                                        'RF, gini, n_estimators = 50, max_depth=50, min_samples_leaf=1, max_leaf_nodes=1562, min_samples_split=8, class_weight=balanced ',
                                        X, y, criterion='gini', n_estimators=50, max_depth=50, min_samples_leaf=1,
                                        max_leaf_nodes=1562, min_samples_split=8, class_weight='balanced',
                                        stratify=stratify, train_size=train_size)
        run_algorithm_rfc_configuration(metrics,
                                        'RF, gini, n_estimators = 90, max_depth=51, min_samples_leaf=1, max_leaf_nodes=684, min_samples_split=6, class_weight=balanced ',
                                        X, y, criterion='gini', n_estimators=90, max_depth=51, min_samples_leaf=1,
                                        max_leaf_nodes=684, min_samples_split=6, class_weight='balanced',
                                        stratify=stratify, train_size=train_size)
        run_algorithm_rfc_configuration(metrics,
                                        'RF, entropy, n_estimators = 50, max_depth=12, min_samples_leaf=3, max_leaf_nodes=682, min_samples_split=2, class_weight=balanced ',
                                        X, y, criterion='entropy', n_estimators=50, max_depth=12, min_samples_leaf=3,
                                        max_leaf_nodes=682, min_samples_split=2, class_weight='balanced',
                                        stratify=stratify, train_size=train_size)
        run_algorithm_rfc_configuration(metrics,
                                        'RF, entropy, n_estimators = 90, max_depth=10, min_samples_leaf=3, max_leaf_nodes=1564, min_samples_split=4, class_weight=balanced  ',
                                        X, y, criterion='entropy', n_estimators=90, max_depth=10, min_samples_leaf=3,
                                        max_leaf_nodes=1564, min_samples_split=4, class_weight='balanced',
                                        stratify=stratify, train_size=train_size)
        run_algorithm_rfc_configuration(metrics,
                                        'RF, entropy, n_estimators = 51, max_depth=11, min_samples_leaf=3, max_leaf_nodes=680, min_samples_split=2, class_weight=balanced ',
                                        X, y, criterion='entropy', n_estimators=51, max_depth=11, min_samples_leaf=3,
                                        max_leaf_nodes=680, min_samples_split=2, class_weight='balanced',
                                        stratify=stratify, train_size=train_size)
        run_algorithm_rfc_configuration(metrics,
                                        'RF, entropy, n_estimators = 100, max_depth=54, min_samples_leaf=3, max_leaf_nodes=1562, min_samples_split=6, class_weight=balanced',
                                        X, y, criterion='entropy', n_estimators=100, max_depth=54, min_samples_leaf=3,
                                        max_leaf_nodes=1562, min_samples_split=6, class_weight='balanced',
                                        stratify=stratify, train_size=train_size)
        run_algorithm_rfc_configuration(metrics,
                                        'RF, entropy, n_estimators = 110, max_depth=53, min_samples_leaf=3, max_leaf_nodes=1564, min_samples_split=4, class_weight=balanced ',
                                        X, y, criterion='entropy', n_estimators=110, max_depth=53, min_samples_leaf=3,
                                        max_leaf_nodes=1564, min_samples_split=4, class_weight='balanced',
                                        stratify=stratify, train_size=train_size)
        run_algorithm_rfc_configuration(metrics,
                                        'RF, gini, n_estimators = 90, max_depth=14, min_samples_leaf=1, max_leaf_nodes=None, min_samples_split=4, class_weight=balanced     ',
                                        X, y, criterion='gini', n_estimators=90, max_depth=14, min_samples_leaf=1,
                                        max_leaf_nodes=None, min_samples_split=4, class_weight='balanced',
                                        stratify=stratify, train_size=train_size)
        run_algorithm_rfc_configuration(metrics,
                                        'RF, entropy, n_estimators = 100, max_depth=11, min_samples_leaf=1, max_leaf_nodes=680, min_samples_split=6, class_weight=balanced  ',
                                        X, y, criterion='entropy', n_estimators=100, max_depth=11, min_samples_leaf=1,
                                        max_leaf_nodes=680, min_samples_split=6, class_weight='balanced',
                                        stratify=stratify, train_size=train_size)
        run_algorithm_rfc_configuration(metrics,
                                        'RF, gini, n_estimators = 100, max_depth=54, min_samples_leaf=1, max_leaf_nodes=1561, min_samples_split=10, class_weight=balanced  ',
                                        X, y, criterion='gini', n_estimators=100, max_depth=54, min_samples_leaf=1,
                                        max_leaf_nodes=1561, min_samples_split=10, class_weight='balanced',
                                        stratify=stratify, train_size=train_size)
        # run_algorithm_rfc_configuration(metrics,
        #                                 '',
        #                                 X, y, criterion='gini', n_estimators=100, max_depth=12, min_samples_leaf=1,
        #                                 max_leaf_nodes=1773, min_samples_split=4, class_weight='balanced',
        #                                 stratify=stratify, train_size=train_size)
        # run_algorithm_rfc_configuration(metrics,
        #                                 '',
        #                                 X, y, criterion='gini', n_estimators=100, max_depth=12, min_samples_leaf=3,
        #                                 max_leaf_nodes=1328, min_samples_split=5, class_weight='balanced',
        #                                 stratify=stratify, train_size=train_size)
        # run_algorithm_rfc_configuration(metrics,
        #                                 '',
        #                                 X, y, criterion='entropy', n_estimators=100, max_depth=12, min_samples_leaf=5,
        #                                 max_leaf_nodes=1080, min_samples_split=3, class_weight='balanced',
        #                                 stratify=stratify, train_size=train_size)
        # run_algorithm_rfc_configuration(metrics,
        #                                 '',
        #                                 X, y, criterion='gini', n_estimators=100, max_depth=12, min_samples_leaf=5,
        #                                 max_leaf_nodes=323, min_samples_split=5, class_weight='balanced',
        #                                 stratify=stratify, train_size=train_size)
        # run_algorithm_rfc_configuration(metrics,
        #                                 '',
        #                                 X, y, criterion='entropy', n_estimators=100, max_depth=12, min_samples_leaf=1,
        #                                 max_leaf_nodes=539, min_samples_split=9, class_weight='balanced',
        #                                 stratify=stratify, train_size=train_size)
        # run_algorithm_rfc_configuration(metrics,
        #                                 '',
        #                                 X, y, criterion='entropy', n_estimators=100, max_depth=12, min_samples_leaf=3,
        #                                 max_leaf_nodes=560, min_samples_split=3, class_weight='balanced',
        #                                 stratify=stratify, train_size=train_size)
        # run_algorithm_rfc_configuration(metrics,
        #                                 '',
        #                                 X, y, criterion='entropy', n_estimators=100, max_depth=12, min_samples_leaf=5,
        #                                 max_leaf_nodes=317, min_samples_split=7, class_weight='balanced',
        #                                 stratify=stratify, train_size=train_size)
        # run_algorithm_rfc_configuration(metrics,
        #                                 '',
        #                                 X, y, criterion='entropy', n_estimators=100, max_depth=12, min_samples_leaf=1,
        #                                 max_leaf_nodes=1328, min_samples_split=5, class_weight='balanced',
        #                                 stratify=stratify, train_size=train_size)
        # run_algorithm_rfc_configuration(metrics,
        #                                 '',
        #                                 X, y, criterion='gini', n_estimators=100, max_depth=12, min_samples_leaf=3,
        #                                 max_leaf_nodes=954, min_samples_split=8, class_weight='balanced',
        #                                 stratify=stratify, train_size=train_size)
        # run_algorithm_rfc_configuration(metrics,
        #                                 '',
        #                                 X, y, criterion='gini', n_estimators=100, max_depth=12, min_samples_leaf=5,
        #                                 max_leaf_nodes=195, min_samples_split=8, class_weight='balanced',
        #                                 stratify=stratify, train_size=train_size)
        # run_algorithm_rfc_configuration(metrics,
        #                                 '',
        #                                 X, y, criterion='gini', n_estimators=100, max_depth=12, min_samples_leaf=5,
        #                                 max_leaf_nodes=201, min_samples_split=7, class_weight='balanced',
        #                                 stratify=stratify, train_size=train_size)
        # run_algorithm_rfc_configuration(metrics,
        #                                 '',
        #                                 X, y, criterion='entropy', n_estimators=100, max_depth=12, min_samples_leaf=3,
        #                                 max_leaf_nodes=539, min_samples_split=6, class_weight='balanced',
        #                                 stratify=stratify, train_size=train_size)
        # run_algorithm_rfc_configuration(metrics,
        #                                 '',
        #                                 X, y, criterion='entropy', n_estimators=100, max_depth=12, min_samples_leaf=5,
        #                                 max_leaf_nodes=35, min_samples_split=9, class_weight='balanced',
        #                                 stratify=stratify, train_size=train_size)
        # run_algorithm_rfc_configuration(metrics,
        #                                 '',
        #                                 X, y, criterion='gini', n_estimators=100, max_depth=12, min_samples_leaf=1,
        #                                 max_leaf_nodes=323, min_samples_split=8, class_weight='balanced',
        #                                 stratify=stratify, train_size=train_size)
        # run_algorithm_rfc_configuration(metrics,
        #                                 '',
        #                                 X, y, criterion='entropy', n_estimators=100, max_depth=12, min_samples_leaf=7,
        #                                 max_leaf_nodes=1080, min_samples_split=9, class_weight='balanced',
        #                                 stratify=stratify, train_size=train_size)
        # run_algorithm_rfc_configuration(metrics,
        #                                 '',
        #                                 X, y, criterion='entropy', n_estimators=100, max_depth=12, min_samples_leaf=3,
        #                                 max_leaf_nodes=562, min_samples_split=8, class_weight='balanced',
        #                                 stratify=stratify, train_size=train_size)
        # run_algorithm_rfc_configuration(metrics,
        #                                 '',
        #                                 X, y, criterion='gini', n_estimators=100, max_depth=12, min_samples_leaf=13,
        #                                 max_leaf_nodes=1300, min_samples_split=7, class_weight='balanced',
        #                                 stratify=stratify, train_size=train_size)
        # run_algorithm_rfc_configuration(metrics,
        #                                 '',
        #                                 X, y, criterion='entropy', n_estimators=100, max_depth=12, min_samples_leaf=7,
        #                                 max_leaf_nodes=1000, min_samples_split=4, class_weight='balanced',
        #                                 stratify=stratify, train_size=train_size)
        # run_algorithm_rfc_configuration(metrics,
        #                                 '',
        #                                 X, y, criterion='entropy', n_estimators=100, max_depth=12, min_samples_leaf=17,
        #                                 max_leaf_nodes=1000, min_samples_split=9, class_weight='balanced',
        #                                 stratify=stratify, train_size=train_size)
        # run_algorithm_rfc_configuration(metrics,
        #                                 '',
        #                                 X, y, criterion='gini', n_estimators=100, max_depth=12, min_samples_leaf=3,
        #                                 max_leaf_nodes=1336, min_samples_split=4, class_weight='balanced',
        #                                 stratify=stratify, train_size=train_size)

    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.groupby(['label'], as_index=False).agg({'precision': 'mean', 'recall': 'mean',
                                                                    'f1_score': 'mean', 'roc_auc': 'mean',
                                                                    'n_estimators': 'first',
                                                                    'bootstrap': 'first', 'oob_score': 'first',
                                                                    'max_samples': 'first',
                                                                    'min_samples_leaf': 'first',
                                                                    'min_samples_split': 'first',
                                                                    'min_weight_fraction_leaf': 'first',
                                                                    'max_features': 'first',
                                                                    'max_leaf_nodes': 'first',
                                                                    'min_impurity_decrease': 'first',
                                                                    'class_weight': 'first',
                                                                    'ccp_alpha': 'first',
                                                                    'max_depth': 'first',
                                                                    'criterion': 'first'})
    metrics_df = compute_average_metric(metrics_df)
    metrics_df.sort_values(by=['average_metric'], ascending=False, inplace=True)
    metrics = appendMetricsTOCSV(my_filename, metrics_df, init_metrics_for_rfc, header=True)


def run_algorithm_rfc_configuration_feature_importance(criterion='gini', n_estimators=100,
                                                       min_samples_leaf=32, min_samples_split=2, max_depth=12,
                                                       max_leaf_nodes=None, max_features='sqrt',
                                                       min_weight_fraction_leaf=0.0,
                                                       min_impurity_decrease=0.0, bootstrap=True, oob_score=False,
                                                       class_weight=None, ccp_alpha=0.0, max_samples=None,
                                                       stratify=False, train_size=0.8,
                                                       normalize_data=False, scaler='min-max'
                                                       ):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    feature_names = X.columns
    print(feature_names)

    X_train, X_test, y_train, y_test = split_data_in_testing_training(X, y, stratify, train_size)

    # Creating the classifier object
    classifier = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                        min_samples_leaf=min_samples_leaf,
                                        min_samples_split=min_samples_split, max_depth=max_depth,
                                        max_leaf_nodes=max_leaf_nodes, max_features=max_features,
                                        min_weight_fraction_leaf=min_weight_fraction_leaf,
                                        min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap,
                                        oob_score=oob_score, class_weight=class_weight, ccp_alpha=ccp_alpha,
                                        max_samples=max_samples, n_jobs=-1)

    # Performing training
    classifier.fit(X_train, y_train)

    start_time = time.time()
    importances = classifier.feature_importances_
    std = np.std([tree.feature_importances_ for tree in classifier.estimators_], axis=0)
    elapsed_time = time.time() - start_time

    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    forest_importances = pd.Series(importances, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    # ax.set_title("Feature importances using MDI (Random Forest)")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

    #     ---------------------------------------------------
    start_time = time.time()
    result = permutation_importance(
        classifier, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
    )
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
    forest_importances = pd.Series(result.importances_mean, index=feature_names)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    # ax.set_title("Feature importances using permutation on full model\n(Random Forest)")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()


def run_algorithm_rfc_parallel(filename='', path='', stratify=False, train_size=0.8,
                               normalize_data=False, scaler='min-max', no_threads=8):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_rfc()
    my_filename = os.path.join(path, 'results/rfc', filename)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_rfc, header=True)

    criterion_list = ['gini', 'entropy']
    n_estimators_list = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240,
                         250, 260, 270, 280, 290, 300]
    max_depth_list = list(chain(range(5, 15), range(45, 55, 2)))
    min_samples_leaf_list = list(chain(range(1, 7), range(11, 22, 2), [32]))
    min_samples_split_list = list(chain(range(2, 11)))
    max_leaf_nodes_list = [None, 680, 681, 682, 683, 684, 1560, 1561, 1562, 1563, 1564]

    # 1003860 configs

    with Manager() as manager:
        q_metrics = manager.Queue()
        jobs = []

        with Pool(no_threads) as pool:
            watcher = pool.apply_async(listener_write_to_file, (q_metrics, my_filename))
            for criterion in criterion_list:
                for n_estimators in n_estimators_list:
                    for max_depth in max_depth_list:
                        for min_samples_leaf in min_samples_leaf_list:
                            for min_samples_split in min_samples_split_list:
                                for max_leaf_nodes in max_leaf_nodes_list:
                                    job = pool.apply_async(run_algorithm_rfc_configuration_parallel,
                                                           (X, y, q_metrics,
                                                            criterion, n_estimators, min_samples_leaf, min_samples_split,
                                                            max_depth, max_leaf_nodes, 'sqrt', 0.0, 0.0, True, False,
                                                            'balanced', 0.0, None,
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

def create_label_for_rfc_for_row(row_rf):
    return create_label_for_rfc(row_rf['criterion'], row_rf['n_estimators'],
                         row_rf['min_samples_leaf'], row_rf['min_samples_split'],
                         row_rf['max_depth'], row_rf['max_leaf_nodes'], row_rf['max_features'],
                         row_rf['min_weight_fraction_leaf'], row_rf['min_impurity_decrease'],
                         row_rf['bootstrap'], row_rf['oob_score'], 'balanced',
                         row_rf['ccp_alpha'], row_rf['max_samples'])

def create_label_for_rfc(criterion, n_estimators, min_samples_leaf, min_samples_split, max_depth,
                         max_leaf_nodes, max_features, min_weight_fraction_leaf,
                         min_impurity_decrease, bootstrap, oob_score,
                         class_weight, ccp_alpha, max_samples):
    return "RF, criterion=" + str(criterion) + ", n_estimators=" + str(n_estimators) + ", min_samples_leaf=" + str(
        min_samples_leaf) + ", min_samples_split=" + str(min_samples_split) + ", max_depth=" + str(
        max_depth) + ", max_leaf_nodes=" + str(max_leaf_nodes) + ", max_features=" + str(
        max_features) + ", min_weight_fraction_leaf" + str(min_weight_fraction_leaf) + ", min_impurity_decrease=" + str(
        min_impurity_decrease) + ", bootstrap=" + str(bootstrap) + ", oob_score=" + str(
        oob_score) + ", class_weight=" + str(class_weight) + ", ccp_alpha=" + str(ccp_alpha) + ", max_samples=" + str(
        max_samples)


def run_algorithm_rfc_configuration_parallel(X, y, q_metrics,
                                             criterion='gini', n_estimators=100,
                                             min_samples_leaf=32, min_samples_split=2, max_depth=12,
                                             max_leaf_nodes=None, max_features='sqrt', min_weight_fraction_leaf=0.0,
                                             min_impurity_decrease=0.0, bootstrap=True, oob_score=False,
                                             class_weight='balanced', ccp_alpha=0.0, max_samples=None,
                                             stratify=False, train_size=0.8):
    X_train, X_test, y_train, y_test = split_data_in_testing_training(X, y, stratify, train_size)
    try:
        # Creating the classifier object
        classifier = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                            min_samples_leaf=min_samples_leaf,
                                            min_samples_split=min_samples_split, max_depth=max_depth,
                                            max_leaf_nodes=max_leaf_nodes, max_features=max_features,
                                            min_weight_fraction_leaf=min_weight_fraction_leaf,
                                            min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap,
                                            oob_score=oob_score, class_weight=class_weight, ccp_alpha=ccp_alpha,
                                            max_samples=max_samples, n_jobs=None)

        # Performing training
        classifier.fit(X_train, y_train)

        # Make predictions
        y_pred, y_pred_probabilities = prediction(X_test, classifier)

        # Compute metrics
        label = create_label_for_rfc(criterion, n_estimators, min_samples_leaf, min_samples_split, max_depth,
                                     max_leaf_nodes, max_features, min_weight_fraction_leaf,
                                     min_impurity_decrease, bootstrap, oob_score,
                                     class_weight, ccp_alpha, max_samples)
        precision, recall, f1, roc_auc = cal_metrics(y_test, y_pred, y_pred_probabilities, label, classifier)
        string_results_for_queue = convert_metrics_to_csv(',', label,
                                                          criterion, n_estimators, min_samples_leaf, min_samples_split,
                                                          max_depth, max_leaf_nodes, max_features,
                                                          min_weight_fraction_leaf, min_impurity_decrease, bootstrap,
                                                          oob_score, class_weight, ccp_alpha, max_samples,
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


def run_algorithm_rfc_configuration(metrics, label, X, y,
                                    criterion='gini', n_estimators=100,
                                    min_samples_leaf=32, min_samples_split=2, max_depth=12,
                                    max_leaf_nodes=None, max_features='sqrt', min_weight_fraction_leaf=0.0,
                                    min_impurity_decrease=0.0, bootstrap=True, oob_score=False,
                                    class_weight='balanced', ccp_alpha=0.0, max_samples=None,
                                    stratify=False, train_size=0.8
                                    ):
    X_train, X_test, y_train, y_test = split_data_in_testing_training(X, y, stratify, train_size)

    # Creating the classifier object
    classifier = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                        min_samples_leaf=min_samples_leaf,
                                        min_samples_split=min_samples_split, max_depth=max_depth,
                                        max_leaf_nodes=max_leaf_nodes, max_features=max_features,
                                        min_weight_fraction_leaf=min_weight_fraction_leaf,
                                        min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap,
                                        oob_score=oob_score, class_weight=class_weight, ccp_alpha=ccp_alpha,
                                        max_samples=max_samples, n_jobs=-1)

    # Performing training
    classifier.fit(X_train, y_train)

    # Make predictions
    y_pred, y_pred_probabilities = prediction(X_test, classifier)

    # Compute metrics
    precision, recall, f1, roc_auc = cal_metrics(y_test, y_pred, y_pred_probabilities, label, classifier)
    metrics['label'].append(label)
    metrics['criterion'].append(criterion)
    metrics['n_estimators'].append(n_estimators)
    metrics['min_samples_leaf'].append(min_samples_leaf)
    metrics['min_samples_split'].append(min_samples_split)
    metrics['max_depth'].append(max_depth)
    metrics['max_leaf_nodes'].append(max_leaf_nodes)
    metrics['max_features'].append(max_features)
    metrics['min_weight_fraction_leaf'].append(min_weight_fraction_leaf)
    metrics['min_impurity_decrease'].append(min_impurity_decrease)
    metrics['bootstrap'].append(bootstrap)
    metrics['oob_score'].append(oob_score)
    metrics['class_weight'].append(class_weight)
    metrics['ccp_alpha'].append(ccp_alpha)
    metrics['max_samples'].append(max_samples)

    metrics['precision'].append(precision)
    metrics['recall'].append(recall)
    metrics['f1_score'].append(f1)
    metrics['roc_auc'].append(roc_auc)


def run_algorithm_rfc_configuration_with_k_fold(X, y,
                                                criterion='gini', n_estimators=100,
                                                min_samples_leaf=32, min_samples_split=2, max_depth=12,
                                                max_leaf_nodes=None, max_features='sqrt', min_weight_fraction_leaf=0.0,
                                                min_impurity_decrease=0.0, bootstrap=True, oob_score=False,
                                                class_weight='balanced', ccp_alpha=0.0, max_samples=None
                                                , n_repeats=10, n_splits=5):
    # Creating the classifier object
    classifier = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                        min_samples_leaf=min_samples_leaf,
                                        min_samples_split=min_samples_split, max_depth=max_depth,
                                        max_leaf_nodes=max_leaf_nodes, max_features=max_features,
                                        min_weight_fraction_leaf=min_weight_fraction_leaf,
                                        min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap,
                                        oob_score=oob_score, class_weight=class_weight, ccp_alpha=ccp_alpha,
                                        max_samples=max_samples, n_jobs=-1)

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=randint(1, 1000000))
    scores = cross_validate(classifier, X, y, scoring=['precision', 'recall', 'f1', 'roc_auc'], cv=rskf, n_jobs=-1,
                            return_train_score=False)
    # report performance
    print(scores.get('test_precision').mean())
    print(scores.get('test_recall').mean())
    print(scores.get('test_f1').mean())
    print(scores.get('test_roc_auc').mean())


def init_metrics_for_rfc():
    return {'label': [],
            'criterion': [], 'n_estimators': [],
            'min_samples_leaf': [], 'min_samples_split': [], 'max_depth': [],
            'max_leaf_nodes': [], 'max_features': [], 'min_weight_fraction_leaf': [],
            'min_impurity_decrease': [], 'bootstrap': [], 'oob_score': [],
            'class_weight': [], 'ccp_alpha': [], 'max_samples': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'roc_auc': []
            }


def run_algorithm_rf(filename='', path='', stratify=False, train_size=0.8,
                     normalize_data=False, scaler='min-max'):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    print('------------------------' + str(len(X)))
    metrics = init_metrics_for_rfc()

    my_filename = os.path.join(path, 'results/rfc', filename)

    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_rfc, header=True)
    # GINI - Equivalent for islam2019mapreduce
    # run_algorithm_rfc_configuration(metrics, 'RF, GINI, n_estimators = 100, min_samples_leaf=32, max_depth = 12',
    #                                 X, y, criterion='gini', stratify=stratify, class_weight='balanced',
    #                                 train_size=train_size)
    # # ENTROPY with the rest configs as for islam2019mapreduce
    # run_algorithm_rfc_configuration(metrics, 'RF, ENTROPY, n_estimators = 100, min_samples_leaf=32, max_depth = 12',
    #                                 X, y, criterion='entropy', class_weight='balanced',
    #                                 stratify=stratify, train_size=train_size)
    # #
    # # # CALIBRATING n_estimators param + ENTROPY/GINI
    # for n_estimators in chain(range(1, 80, 10), range(80, 120, 5), range(120, 250, 10), range(250, 500, 25), range(500, 1000, 50)):
    #   run_algorithm_rfc_configuration(metrics, 'RF, GINI, n_estimators = '+str(n_estimators)+
    #                                   ', min_samples_leaf=32, max_depth = 12',
    #                                   X, y, criterion = 'gini', n_estimators=n_estimators,
    #                                   class_weight='balanced',
    #                                   stratify = stratify, train_size = train_size)
    #   run_algorithm_rfc_configuration(metrics, 'RF, ENTROPY, n_estimators = '+str(n_estimators)+
    #                                   ', min_samples_leaf=32, max_depth = 12', X, y,
    #                                   criterion = 'entropy', n_estimators = n_estimators,
    #                                   class_weight='balanced',
    #                                   stratify = stratify, train_size = train_size)
    #
    # # CALIBRATING min_samples_leaf param + ENTROPY/GINI
    # for min_samples_leaf in range(1, 100, 5):
    #     run_algorithm_rfc_configuration(metrics, 'RF, GINI, n_estimators = 100, min_samples_leaf=' +
    #                                     str(min_samples_leaf) + ', max_depth = 12', X, y, criterion='gini',
    #                                     min_samples_leaf=min_samples_leaf, stratify=stratify,
    #                                     class_weight='balanced',
    #                                     train_size=train_size)
    #     run_algorithm_rfc_configuration(metrics, 'RF, ENTROPY, n_estimators = 100, min_samples_leaf=' +
    #                                     str(min_samples_leaf) + ', max_depth = 12', X, y, criterion='entropy',
    #                                     min_samples_leaf=min_samples_leaf, stratify=stratify,
    #                                     class_weight='balanced',
    #                                     train_size=train_size)
    #
    # # CALIBRATING max_depth param + GINI/ENTROPY
    # run_algorithm_rfc_configuration(metrics, 'RF, GINI, n_estimators = 100, min_samples_leaf=32, max_depth = None',
    #                                 X, y, criterion='gini', stratify=stratify,
    #                                 class_weight='balanced',
    #                                 train_size=train_size)
    # run_algorithm_rfc_configuration(metrics, 'RF, ENTROPY, n_estimators = 100, min_samples_leaf=32, max_depth = None',
    #                                 X, y, criterion='entropy',  stratify=stratify, class_weight='balanced',
    #                                 train_size=train_size)
    # for max_depth in chain(range(1, 16, 1), range(16, 100, 2)):
    #     run_algorithm_rfc_configuration(metrics, 'RF, GINI, n_estimators = 100, min_samples_leaf=32, max_depth = ' +
    #                                     str(max_depth), X, y, criterion='gini', max_depth=max_depth, class_weight='balanced',
    #                                     stratify=stratify, train_size=train_size)
    #     run_algorithm_rfc_configuration(metrics, 'RF, ENTROPY, n_estimators = 100, min_samples_leaf=32, max_depth = ' +
    #                                     str(max_depth), X, y, criterion='entropy', max_depth=max_depth, class_weight='balanced',
    #                                     stratify=stratify, train_size=train_size)
    #
    # # CALIBRATING min_samples_split param + ENTROPY/GINI
    # for min_samples_split in range(3, len(X), 20):
    #     run_algorithm_rfc_configuration(metrics,
    #                                     'RF, GINI, n_estimators = 100, min_samples_leaf=32, max_depth = 12, min_samples_split = ' +
    #                                     str(min_samples_split), X, y, criterion='gini', class_weight='balanced',
    #                                     min_samples_split=min_samples_split, stratify=stratify,
    #                                     train_size=train_size)
    #     run_algorithm_rfc_configuration(metrics,
    #                                     'RF, ENTROPY, n_estimators = 100, min_samples_leaf=32, max_depth = 12, min_samples_split= ' +
    #                                     str(min_samples_split), X, y, criterion='entropy', class_weight='balanced',
    #                                     min_samples_split=min_samples_split, stratify=stratify,
    #                                     train_size=train_size)
    #
    # # CALIBRATING min_weight_fraction_leaf param + ENTROPY/GINI
    # for min_weight_fraction_leaf in [0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5]:
    #   run_algorithm_rfc_configuration(metrics, 'RF, GINI, n_estimators = 100, min_samples_leaf=32, max_depth = 12, min_weight_fraction_leaf = '+ str(min_weight_fraction_leaf), X, y, criterion = 'gini',
    #                                   min_weight_fraction_leaf=min_weight_fraction_leaf, class_weight='balanced',
    #                                   stratify = stratify, train_size = train_size)
    #   run_algorithm_rfc_configuration(metrics, 'RF, ENTROPY, n_estimators = 100, min_samples_leaf=32, max_depth = 12, min_weight_fraction_leaf= '+str(min_weight_fraction_leaf),
    #                                   X, y, criterion = 'entropy', min_weight_fraction_leaf = min_weight_fraction_leaf,
    #                                   class_weight='balanced', stratify = stratify, train_size = train_size)
    #
    # # CALIBRATING max_features param + ENTROPY/GINI
    # for max_features in range(1, 20, 2):
    #   run_algorithm_rfc_configuration(metrics, 'RF, GINI, n_estimators = 100, min_samples_leaf=32, max_depth = 12, max_features = '+ str(max_features),
    #                                   X, y, criterion = 'gini', max_features=max_features, stratify = stratify,
    #                                   class_weight='balanced', train_size = train_size)
    #   run_algorithm_rfc_configuration(metrics, 'RF, ENTROPY, n_estimators = 100, min_samples_leaf=32, max_depth = 12, max_features= '+str(max_features),
    #                                   X, y, criterion = 'entropy', max_features = max_features, stratify = stratify,
    #                                   class_weight='balanced', train_size = train_size)
    # run_algorithm_rfc_configuration(metrics,
    #                                 'RF, GINI, n_estimators = 100, min_samples_leaf=32, max_depth = 12, max_features = log2',
    #                                 X, y, criterion='gini', max_features='log2', stratify=stratify,
    #                                 class_weight='balanced', train_size=train_size)
    # run_algorithm_rfc_configuration(metrics,
    #                                 'RF, ENTROPY, n_estimators = 100, min_samples_leaf=32, max_depth = 12, max_features= log2',
    #                                 X, y, criterion='entropy', max_features='log2', stratify=stratify,
    #                                 class_weight='balanced', train_size=train_size)
    # run_algorithm_rfc_configuration(metrics,
    #                                 'RF, GINI, n_estimators = 100, min_samples_leaf=32, max_depth = 12, max_features = sqrt',
    #                                 X, y, criterion='gini', max_features='sqrt', stratify=stratify,
    #                                 class_weight='balanced', train_size=train_size)
    # run_algorithm_rfc_configuration(metrics,
    #                                 'RF, ENTROPY, n_estimators = 100, min_samples_leaf=32, max_depth = 12, max_features= sqrt',
    #                                 X, y, criterion='entropy', max_features='sqrt',
    #                                 class_weight='balanced', stratify=stratify, train_size=train_size)
    #
    # # CALIBRATING max_leaf_nodes param + ENTROPY/GINI
    # for max_leaf_nodes in range(2, len(X), 20):
    #     run_algorithm_rfc_configuration(metrics,
    #                                     'RF, GINI, n_estimators = 100, min_samples_leaf=32, max_depth = 12, max_leaf_nodes = ' + str(
    #                                         max_leaf_nodes),
    #                                     X, y, criterion='gini', max_leaf_nodes=max_leaf_nodes,
    #                                     class_weight='balanced', stratify=stratify, train_size=train_size)
    #     run_algorithm_rfc_configuration(metrics,
    #                                     'RF, ENTROPY, n_estimators = 100, min_samples_leaf=32, max_depth = 12, max_leaf_nodes= ' + str(
    #                                         max_leaf_nodes),
    #                                     X, y, criterion='entropy', max_leaf_nodes=max_leaf_nodes,
    #                                     class_weight='balanced', stratify=stratify, train_size=train_size)
    # metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_rfc)
    #
    # # CALIBRATING min_impurity_decrease param + ENTROPY/GINI
    # for min_impurity_decrease in range(0, len(X), 20):
    #   run_algorithm_rfc_configuration(metrics, 'RF, GINI, n_estimators = 100, min_samples_leaf=32, max_depth = 12, min_impurity_decrease = '+ str(min_impurity_decrease),
    #                                   X, y, criterion = 'gini', min_impurity_decrease=min_impurity_decrease,
    #                                   class_weight='balanced', stratify = stratify, train_size = train_size)
    #   run_algorithm_rfc_configuration(metrics, 'RF, ENTROPY, n_estimators = 100, min_samples_leaf=32, max_depth = 12, min_impurity_decrease= '+str(min_impurity_decrease),
    #                                   X, y, criterion = 'entropy', min_impurity_decrease = min_impurity_decrease,
    #                                   class_weight='balanced',stratify = stratify, train_size = train_size)
    # metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_rfc)
    #
    # # CALIBRATING class_weight param + ENTROPY/GINI
    # run_algorithm_rfc_configuration(metrics,
    #                                 'RF, GINI, n_estimators = 100, min_samples_leaf=32, max_depth = 12, class_weight = balanced',
    #                                 X, y, criterion='gini', class_weight='balanced',
    #                                 stratify=stratify, train_size=train_size)
    # run_algorithm_rfc_configuration(metrics,
    #                                 'RF, ENTROPY, n_estimators = 100, min_samples_leaf=32, max_depth = 12, class_weight= balanced',
    #                                 X, y, criterion='entropy', class_weight='balanced',
    #                                 stratify=stratify, train_size=train_size)
    # run_algorithm_rfc_configuration(metrics,
    #                                 'RF, GINI, n_estimators = 100, min_samples_leaf=32, max_depth = 12, class_weight = balanced_subsample',
    #                                 X, y, criterion='gini', class_weight='balanced_subsample', stratify=stratify,
    #                                 train_size=train_size)
    # run_algorithm_rfc_configuration(metrics,
    #                                 'RF, ENTROPY, n_estimators = 100, min_samples_leaf=32, max_depth = 12, class_weight= balanced_subsample',
    #                                 X, y, criterion='entropy', class_weight='balanced_subsample',
    #                                 stratify=stratify, train_size=train_size)
    #
    # # CALIBRATING bootstrap param + ENTROPY/GINI
    # run_algorithm_rfc_configuration(metrics,
    #                                 'RF, GINI, n_estimators = 100, min_samples_leaf=32, max_depth = 12, bootstrap = True',
    #                                 X, y, criterion='gini', class_weight='balanced', bootstrap=True, stratify=stratify,
    #                                 train_size=train_size)
    # run_algorithm_rfc_configuration(metrics,
    #                                 'RF, ENTROPY, n_estimators = 100, min_samples_leaf=32, max_depth = 12, bootstrap= True',
    #                                 X, y, criterion='entropy', bootstrap=True, stratify=stratify,
    #                                 class_weight='balanced',train_size=train_size)
    # run_algorithm_rfc_configuration(metrics,
    #                                 'RF, GINI, n_estimators = 100, min_samples_leaf=32, max_depth = 12, bootstrap = False',
    #                                 X, y, criterion='gini', bootstrap=False, stratify=stratify,
    #                                 class_weight='balanced',train_size=train_size)
    # run_algorithm_rfc_configuration(metrics,
    #                                 'RF, ENTROPY, n_estimators = 100, min_samples_leaf=32, max_depth = 12, bootstrap= False',
    #                                 X, y, criterion='entropy', bootstrap=False, stratify=stratify,
    #                                 class_weight='balanced', train_size=train_size)
    #
    # # CALIBRATING oob_score param + ENTROPY/GINI
    # run_algorithm_rfc_configuration(metrics,
    #                                 'RF, GINI, n_estimators = 100, min_samples_leaf=32, max_depth = 12, bootstrap = True, oob_score = True',
    #                                 X, y, criterion='gini', bootstrap=True, oob_score=True,
    #                                 class_weight='balanced', stratify=stratify, train_size=train_size)
    # run_algorithm_rfc_configuration(metrics,
    #                                 'RF, ENTROPY, n_estimators = 100, min_samples_leaf=32, max_depth = 12, bootstrap= True, oob_score = True',
    #                                 X, y, criterion='entropy', bootstrap=True, oob_score=True,
    #                                 class_weight='balanced', stratify=stratify, train_size=train_size)
    # run_algorithm_rfc_configuration(metrics,
    #                                 'RF, GINI, n_estimators = 100, min_samples_leaf=32, max_depth = 12, bootstrap = True, oob_score = False',
    #                                 X, y, criterion='gini', bootstrap=True, oob_score=False,
    #                                 class_weight='balanced',stratify=stratify, train_size=train_size)
    # run_algorithm_rfc_configuration(metrics,
    #                                 'RF, ENTROPY, n_estimators = 100, min_samples_leaf=32, max_depth = 12, bootstrap= True, oob_score = False',
    #                                 X, y, criterion='entropy', bootstrap=True, oob_score=False,
    #                                 class_weight='balanced', stratify=stratify, train_size=train_size)
    #
    # # CALIBRATING ccp_alpha param + ENTROPY/GINI
    # for ccp_alpha in [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1]:
    #   run_algorithm_rfc_configuration(metrics, 'RF, GINI, n_estimators = 100, min_samples_leaf=32, max_depth = 12, ccp_alpha = '+
    #                                   str(ccp_alpha), X, y, criterion = 'gini', ccp_alpha=ccp_alpha,
    #                                   class_weight='balanced', stratify = stratify, train_size = train_size)
    #   run_algorithm_rfc_configuration(metrics, 'RF, ENTROPY, n_estimators = 100, min_samples_leaf=32, max_depth = 12, ccp_alpha= '+
    #                                   str(ccp_alpha), X, y, criterion = 'entropy', ccp_alpha = ccp_alpha,
    #                                   class_weight='balanced', stratify = stratify, train_size = train_size)
    #
    # # CALIBRATING max_samples param + ENTROPY/GINI
    # for max_samples in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]:
    #   run_algorithm_rfc_configuration(metrics, 'RF, GINI, n_estimators = 100, min_samples_leaf=32, max_depth = 12, max_samples = '+
    #                                   str(max_samples), X, y, criterion = 'gini', max_samples=max_samples,
    #                                   class_weight='balanced', stratify = stratify, train_size = train_size)
    #   run_algorithm_rfc_configuration(metrics, 'RF, ENTROPY, n_estimators = 100, min_samples_leaf=32, max_depth = 12, max_samples= '+
    #                                   str(max_samples), X, y, criterion = 'entropy', max_samples = max_samples,
    #                                   class_weight='balanced', stratify = stratify, train_size = train_size)
    #
    #   metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_rfc)
    #

    for criterion in ['gini', 'entropy']:
        for n_estimators in [50, 51, 90, 100, 110]:
            concatenated_max_depth = chain(range(10, 15), range(50, 55))
            for max_depth in concatenated_max_depth:
                concatenated_min_samples_leaf = chain(
                    range(1, 7, 2), range(11, 22, 2), range(32, 33)
                )
                for min_samples_leaf in concatenated_min_samples_leaf:
                    for min_samples_split in range(2, 11, 2):
                        for max_leaf_nodes in [None, 680, 681, 682, 683, 684, 1560, 1561, 1562, 1563, 1564]:
                            run_algorithm_rfc_configuration(metrics, 'RF, ' + criterion + ', n_estimators = ' +
                                                            str(n_estimators) + ', max_depth=' + str(max_depth) +
                                                            ', min_samples_leaf=' + str(min_samples_leaf) +
                                                            ', max_leaf_nodes=' + str(
                                max_leaf_nodes) + ', min_samples_split=' +
                                                            str(min_samples_split) + ', class_weight=balanced', X, y,
                                                            criterion=criterion,
                                                            n_estimators=n_estimators,
                                                            max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                                            max_leaf_nodes=max_leaf_nodes,
                                                            min_samples_split=min_samples_split,
                                                            class_weight='balanced',
                                                            stratify=stratify, train_size=train_size)
                        metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_rfc)

    # # concatenated_n_estimators = chain(range(76, 108, 2), range(155,175,2), range(915, 925, 2))
    # for n_estimators in range(100, 101):
    # # for n_estimators in concatenated_n_estimators:
    # #     concatenated_max_depth = chain(range(12, 13), range(30, 35))
    #     for max_depth in range(12, 13):
    #     # for max_depth in concatenated_max_depth:
    #         # 11+5=15
    #         concatenated_min_samples_leaf = chain(range(1, 20, 2), range(30, 35))
    #         for min_samples_leaf in concatenated_min_samples_leaf:
    #             concatenated_max_leaf_nodes = chain(range(35, 45, 2), range(150, 155, 2),
    #                                                 range(1110, 1120, 2), range(1326, 1337, 2),
    #                                                 range(1475, 1485, 2), range(560, 570, 2),
    #                                                 range(1300, 1310, 2), range(1765, 1775, 2),
    #                                                 range(315, 325, 2), range(950, 960, 2),
    #                                                 range(1080, 1090, 2), range(75, 85, 2),
    #                                                 range(195, 205, 2), range(1000, 1015, 2),
    #                                                 range(537, 547, 2))
    #             # 5+3+5+5+5+5+5+5+5+5+5+5+5+3+5 = 13*5+6=65+6=71
    #             for max_leaf_nodes in concatenated_max_leaf_nodes:
    #                 concatenated_min_samples_split = chain(range(3, 10), range(46, 56, 2))
    #                 # 7 + 5 = 12
    #                 for min_samples_split in concatenated_min_samples_split:
    #                     run_algorithm_rfc_configuration(metrics, 'RF, GINI, n_estimators = ' +
    #                                                     str(n_estimators) + ', max_depth=' + str(max_depth) +
    #                                                     ', min_samples_leaf=' + str(min_samples_leaf) +
    #                                                     ', max_leaf_nodes=' + str(
    #                         max_leaf_nodes) + ', min_samples_split=' +
    #                                                     str(min_samples_split) + ', class_weight=balanced', X, y,
    #                                                     criterion='gini',
    #                                                     n_estimators=n_estimators,
    #                                                     max_depth=max_depth, min_samples_leaf=min_samples_leaf,
    #                                                     max_leaf_nodes=max_leaf_nodes,
    #                                                     min_samples_split=min_samples_split, class_weight='balanced',
    #                                                     stratify=stratify, train_size=train_size)
    #                     run_algorithm_rfc_configuration(metrics, 'RF, ENTROPY, n_estimators = ' +
    #                                                     str(n_estimators) + ', max_depth=' + str(max_depth) +
    #                                                     ', min_samples_leaf=' + str(min_samples_leaf) +
    #                                                     ', max_leaf_nodes=' + str(
    #                         max_leaf_nodes) + ', min_samples_split=' +
    #                                                     str(min_samples_split) + ', class_weight = balanced', X, y,
    #                                                     criterion='entropy',
    #                                                     n_estimators=n_estimators,
    #                                                     max_depth=max_depth, min_samples_leaf=min_samples_leaf,
    #                                                     max_leaf_nodes=max_leaf_nodes,
    #                                                     min_samples_split=min_samples_split, class_weight='balanced',
    #                                                     stratify=stratify, train_size=train_size)
    #                 metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_rfc)

    # export metrics to CSV FILE
    # df_metrics = pd.DataFrame(metrics)
    # df_metrics.to_csv(my_filename, encoding='utf-8', index= True)
    # df_metrics = pd.DataFrame(metrics)
    # df_metrics.to_csv('/content/drive/MyDrive/code/'+filename, encoding='utf-8', index= True)


def run_algorithm_rf_with_k_fold(normalize_data=False, scaler='min-max'):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    run_algorithm_rfc_configuration_with_k_fold(X, y, criterion='entropy', n_estimators=110, max_depth=53,
                                                min_samples_leaf=3, max_leaf_nodes=1564, min_samples_split=4,
                                                class_weight='balanced',
                                                n_splits=2, n_repeats=10)
