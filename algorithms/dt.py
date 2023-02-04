import os

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from data_pre import load_normalized_dataset, split_data_in_testing_training
from utils import split_data, cal_metrics, prediction, appendMetricsTOCSV
import time
import matplotlib.pyplot as plt
from itertools import chain



def run_top_20_DT_configs(filename='', path = '', stratify=True, train_size=0.8,
                           normalize_data=True, scaler='min-max', n_rep=100):
    y, X = load_normalized_dataset(file = None, normalize = normalize_data, scaler=scaler)
    metrics = init_metrics_for_DT()

    # full_path_filename = '/content/drive/MyDrive/code/' + filename
    # path_to_script = os.path.dirname(os.path.abspath(__file__))
    my_filename = os.path.join(path, 'results/dt', filename)

    for i in range(1, n_rep):
        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 11, max_depth = 11, max_leaf_nodes = 40, min_samples_split = 2, class_weight = balanced',
                                       X, y, criterion='entropy', min_samples_leaf = 11, max_depth=11,
                                       max_leaf_nodes=40, min_samples_split = 2, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 12, max_depth = 10, max_leaf_nodes = 40, min_samples_split = 20, class_weight = balanced',
                                       X, y, criterion='entropy', min_samples_leaf = 12, max_depth=10,
                                       max_leaf_nodes=40, min_samples_split = 20, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 13, max_depth = 20, max_leaf_nodes = 70, min_samples_split = 20, class_weight = balanced',
                                       X, y, criterion='entropy', min_samples_leaf = 13, max_depth=20,
                                       max_leaf_nodes=70, min_samples_split = 20, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 13, max_depth = 87, max_leaf_nodes = 30, min_samples_split = 6, class_weight = balanced',
                                       X, y, criterion='entropy', min_samples_leaf = 13, max_depth=87,
                                       max_leaf_nodes=30, min_samples_split = 6, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, GINI, min_samples_leaf=11, max_depth = 89, max_leaf_nodes = 80, min_samples_split=8, class_weight = balanced',
                                       X, y, criterion='gini', min_samples_leaf = 11, max_depth=89,
                                       max_leaf_nodes=80, min_samples_split = 8, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 12, max_depth = 20, max_leaf_nodes = 40, min_samples_split = 8, class_weight = balanced',
                                       X, y, criterion='entropy', min_samples_leaf = 12, max_depth=20,
                                       max_leaf_nodes=40, min_samples_split = 8, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 11, max_depth = 30, max_leaf_nodes = 60, min_samples_split = 20, class_weight = balanced',
                                       X, y, criterion='entropy', min_samples_leaf = 11, max_depth=30,
                                       max_leaf_nodes=60, min_samples_split = 20, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 14, max_depth = 25, max_leaf_nodes = 70, min_samples_split = 2, class_weight = balanced',
                                       X, y, criterion='entropy', min_samples_leaf = 14, max_depth=25,
                                       max_leaf_nodes=70, min_samples_split = 2, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, GINI, min_samples_leaf=12, max_depth = 87, max_leaf_nodes = 70, min_samples_split=4, class_weight = balanced',
                                       X, y, criterion='gini', min_samples_leaf = 12, max_depth=87,
                                       max_leaf_nodes=70, min_samples_split = 4, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 10, max_depth = 85, max_leaf_nodes = 50, min_samples_split = 2, class_weight = balanced',
                                       X, y, criterion='entropy', min_samples_leaf = 10, max_depth=85,
                                       max_leaf_nodes=50, min_samples_split = 2, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 14, max_depth = 87, max_leaf_nodes = 80, min_samples_split = 4, class_weight = balanced',
                                       X, y, criterion='entropy', min_samples_leaf = 14, max_depth=87,
                                       max_leaf_nodes=80, min_samples_split = 4, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 13, max_depth = 12, max_leaf_nodes = 30, min_samples_split = 20, class_weight = balanced',
                                       X, y, criterion='entropy', min_samples_leaf = 13, max_depth=12,
                                       max_leaf_nodes=30, min_samples_split = 20, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)

        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 10, max_depth = 14, max_leaf_nodes = 60, min_samples_split = 8, class_weight = balanced',
                                       X, y, criterion='entropy', min_samples_leaf = 10, max_depth=14,
                                       max_leaf_nodes=60, min_samples_split = 8, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 12, max_depth = 87, max_leaf_nodes = 30, min_samples_split = 2, class_weight = balanced',
                                       X, y, criterion='entropy', min_samples_leaf = 12, max_depth=87,
                                       max_leaf_nodes=30, min_samples_split = 2, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, GINI, min_samples_leaf=13, max_depth = 87, max_leaf_nodes = 70, min_samples_split=30, class_weight = balanced',
                                       X, y, criterion='gini', min_samples_leaf = 13, max_depth=87,
                                       max_leaf_nodes=70, min_samples_split = 30, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 10, max_depth = 12, max_leaf_nodes = 40, min_samples_split = 8, class_weight = balanced',
                                       X, y, criterion='entropy', min_samples_leaf = 10, max_depth=12,
                                       max_leaf_nodes=40, min_samples_split = 8, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 10, max_depth = 14, max_leaf_nodes = 30, min_samples_split = 20, class_weight = balanced',
                                       X, y, criterion='entropy', min_samples_leaf = 10, max_depth=14,
                                       max_leaf_nodes=30, min_samples_split = 20, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, GINI, min_samples_leaf=10, max_depth = 35, max_leaf_nodes = 30, min_samples_split=6, class_weight = balanced',
                                       X, y, criterion='gini', min_samples_leaf = 10, max_depth=35,
                                       max_leaf_nodes=30, min_samples_split = 6, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 10, max_depth = 30, max_leaf_nodes = 80, min_samples_split = 8, class_weight = balanced',
                                       X, y, criterion='entropy', min_samples_leaf = 10, max_depth=30,
                                       max_leaf_nodes=80, min_samples_split = 8, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 11, max_depth = 20, max_leaf_nodes = 40, min_samples_split = 20, class_weight = balanced',
                                       X, y, criterion='entropy', min_samples_leaf = 11, max_depth=20,
                                       max_leaf_nodes=40, min_samples_split = 20, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)


        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 11, max_depth = 10, max_leaf_nodes = 70, min_samples_split = 4, class_weight = balanced',
                                       X, y, criterion='entropy', min_samples_leaf = 11, max_depth=10,
                                       max_leaf_nodes=70, min_samples_split = 4, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 13, max_depth = 12, max_leaf_nodes = 30, min_samples_split = 8, class_weight = balanced',
                                       X, y, criterion='entropy', min_samples_leaf = 13, max_depth=12,
                                       max_leaf_nodes=30, min_samples_split = 8, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 13, max_depth = 87, max_leaf_nodes = 70, min_samples_split = 10, class_weight = balanced',
                                       X, y, criterion='entropy', min_samples_leaf = 13, max_depth=87,
                                       max_leaf_nodes=70, min_samples_split = 10, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 14, max_depth = 12, max_leaf_nodes = 40, min_samples_split = 20, class_weight = balanced',
                                       X, y, criterion='entropy', min_samples_leaf = 14, max_depth=12,
                                       max_leaf_nodes=40, min_samples_split = 20, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 11, max_depth = 89, max_leaf_nodes = 40, min_samples_split = 6, class_weight = balanced',
                                       X, y, criterion='entropy', min_samples_leaf = 11, max_depth=89,
                                       max_leaf_nodes=40, min_samples_split = 6, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 11, max_depth = 14, max_leaf_nodes = 60, min_samples_split = 4, class_weight = balanced',
                                       X, y, criterion='entropy', min_samples_leaf = 11, max_depth=14,
                                       max_leaf_nodes=60, min_samples_split = 4, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 10, max_depth = 25, max_leaf_nodes = 80, min_samples_split = 10, class_weight = balanced',
                                       X, y, criterion='entropy', min_samples_leaf = 10, max_depth=25,
                                       max_leaf_nodes=80, min_samples_split = 10, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, GINI, min_samples_leaf=10, max_depth = 85, max_leaf_nodes = 50, min_samples_split=6, class_weight = balanced',
                                       X, y, criterion='gini', min_samples_leaf = 10, max_depth=85,
                                       max_leaf_nodes=50, min_samples_split = 6, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 12, max_depth = 20, max_leaf_nodes = 80, min_samples_split = 6, class_weight = balanced',
                                       X, y, criterion='entropy', min_samples_leaf = 12, max_depth=20,
                                       max_leaf_nodes=80, min_samples_split = 6, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 13, max_depth = 11, max_leaf_nodes = 80, min_samples_split = 4, class_weight = balanced',
                                       X, y, criterion='entropy', min_samples_leaf = 13, max_depth=11,
                                       max_leaf_nodes=80, min_samples_split = 4, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 11, max_depth = 25, max_leaf_nodes = 60, min_samples_split = 40, class_weight = balanced',
                                       X, y, criterion='entropy', min_samples_leaf = 11, max_depth=25,
                                       max_leaf_nodes=60, min_samples_split = 40, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, GINI, min_samples_leaf=10, max_depth = 85, max_leaf_nodes = 50, min_samples_split=8, class_weight = balanced',
                                       X, y, criterion='gini', min_samples_leaf = 10, max_depth=85,
                                       max_leaf_nodes=50, min_samples_split = 8, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 11, max_depth = 11, max_leaf_nodes = 40, min_samples_split = 20, class_weight = balanced',
                                       X, y, criterion='entropy', min_samples_leaf = 11, max_depth=11,
                                       max_leaf_nodes=40, min_samples_split = 20, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 10, max_depth = 91, max_leaf_nodes = 50, min_samples_split = 10, class_weight = balanced',
                                       X, y, criterion='entropy', min_samples_leaf = 10, max_depth=91,
                                       max_leaf_nodes=50, min_samples_split = 10, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, GINI, min_samples_leaf=11, max_depth = 91, max_leaf_nodes = 20, min_samples_split=80, class_weight = balanced',
                                       X, y, criterion='gini', min_samples_leaf = 11, max_depth=91,
                                       max_leaf_nodes=20, min_samples_split = 80, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 14, max_depth = 20, max_leaf_nodes = 40, min_samples_split = 20, class_weight = balanced',
                                       X, y, criterion='entropy', min_samples_leaf = 14, max_depth=20,
                                       max_leaf_nodes=40, min_samples_split = 20, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 11, max_depth = 93, max_leaf_nodes = 50, min_samples_split = 4, class_weight = balanced',
                                       X, y, criterion='entropy', min_samples_leaf = 11, max_depth=93,
                                       max_leaf_nodes=50, min_samples_split = 4, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 12, max_depth = 35, max_leaf_nodes = 80, min_samples_split = 30, class_weight = balanced',
                                       X, y, criterion='entropy', min_samples_leaf = 12, max_depth=35,
                                       max_leaf_nodes=80, min_samples_split = 30, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 12, max_depth = 85, max_leaf_nodes = 40, min_samples_split = 40, class_weight = balanced',
                                       X, y, criterion='entropy', min_samples_leaf = 12, max_depth=85,
                                       max_leaf_nodes=40, min_samples_split = 40, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 12, max_depth = 35, max_leaf_nodes = 40, min_samples_split = 20, class_weight = balanced',
                                       X, y, criterion='entropy', min_samples_leaf = 12, max_depth=35,
                                       max_leaf_nodes=40, min_samples_split = 20, class_weight = 'balanced',
                                       stratify=stratify, train_size=train_size)


    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.groupby(['label'], as_index=False).agg({'precision': 'mean', 'recall': 'mean',
                                                            'f1_score':'mean', 'roc_auc': 'mean',
                                                            'min_samples_leaf': 'first',
                                                            'min_samples_split': 'first',
                                                            'min_weight_fraction_leaf': 'first',
                                                            'max_features': 'first',
                                                            'max_leaf_nodes': 'first',
                                                            'min_impurity_decrease': 'first',
                                                            'class_weight': 'first',
                                                            'ccp_alpha': 'first',
                                                            'splitter':'first', 'max_depth': 'first', 'criterion': 'first'})
    metrics = appendMetricsTOCSV(my_filename, metrics_df, init_metrics_for_DT, header=True)

def run_algorithm_dt_configuration_feature_importance(metrics, label, X, y, criterion='gini',
                                                      splitter='best', max_depth=12, min_samples_leaf=32,
                                                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                                                      max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
                                                      class_weight=None, ccp_alpha=0.0,
                                                      train_size=0.8, stratify=False,
                                                      normalize_data=False, scaler='min-max', feature_names=[]
                                                      ):
    X_train, X_test, y_train, y_test = split_data(X, y, normalize_data=normalize_data, stratify=stratify,
                                                  train_size=train_size, scaler=scaler);

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
    ax.set_title("Feature importances using MDI (Decision Tree)")
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
    ax.set_title("Feature importances using permutation on full model\n(Decision Tree)")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()


def run_algorithm_dt_configuration(metrics, label, X, y, criterion='gini',
                                   splitter='best', max_depth=12, min_samples_leaf=32,
                                   min_samples_split=2, min_weight_fraction_leaf=0.0,
                                   max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
                                   class_weight=None, ccp_alpha=0.0,
                                   train_size=0.8, stratify=False
                                   ):
    X_test, X_train, y_test, y_train = split_data_in_testing_training(X, y, stratify, train_size)

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
    precision, recall, f1, roc_auc = cal_metrics(y_test, y_pred, y_pred_probabilities, label)
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

def run_algorithm_dt(filename='', path = '', stratify=False, train_size=0.8,
                     normalize_data=False, scaler='min-max'):
    y, X = load_normalized_dataset(file = None, normalize = normalize_data, scaler=scaler)
    metrics = init_metrics_for_DT()

    # path_to_script = os.path.dirname(os.path.abspath(__file__))
    my_filename = os.path.join(path, 'results/dt', filename)

    # GINI - Equivalent for islam2019mapreduce
    run_algorithm_dt_configuration(metrics, 'DT, GINI, min_samples_leaf=32, max_depth = 12', X, y, criterion='gini',
                                   stratify=stratify, train_size=train_size)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_DT, header=True)
    # ENTROPY with the rest configs as for islam2019mapreduce
    run_algorithm_dt_configuration(metrics, 'DT, ENTROPY, min_samples_leaf=32, max_depth = 12', X, y,
                                   criterion='entropy', stratify=stratify, train_size=train_size)

    # CALIBRATING min_samples_leaf param + ENTROPY/GINI
    for min_samples_leaf in range(1, 100):
        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf=' + str(min_samples_leaf) + ', max_depth = 12', X,
                                       y, criterion='entropy', min_samples_leaf=min_samples_leaf, stratify=stratify,
                                       train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, GINI, min_samples_leaf=' + str(min_samples_leaf) + ', max_depth = 12', X, y,
                                       criterion='gini', min_samples_leaf=min_samples_leaf, stratify=stratify,
                                       train_size=train_size)

    # CALIBRATING max_depth param + GINI/ENTROPY
    for max_depth in range(1, 100):
        run_algorithm_dt_configuration(metrics, 'DT, ENTROPY, min_samples_leaf= 32, max_depth = ' + str(max_depth), X,
                                       y, criterion='entropy', max_depth=max_depth, stratify=stratify,
                                       train_size=train_size)
        run_algorithm_dt_configuration(metrics, 'DT, GINI, min_samples_leaf=32, max_depth = ' + str(max_depth), X, y,
                                       criterion='gini', max_depth=max_depth, stratify=stratify, train_size=train_size)

    # CALIBRATING min_samples_split param + ENTROPY/GINI
    for min_samples_split in range(2, len(X), 20):
        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 32, max_depth = 12, min_samples_split = ' + str(
                                           min_samples_split), X, y, criterion='entropy',
                                       min_samples_split=min_samples_split, stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, GINI, min_samples_leaf=32, max_depth =12, min_samples_split = ' + str(
                                           min_samples_split), X, y, criterion='gini',
                                       min_samples_split=min_samples_split, stratify=stratify, train_size=train_size)

    # CALIBRATING min_weight_fraction_leaf param + ENTROPY/GINI
    for min_weight_fraction_leaf in [0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5]:
        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 32, max_depth = 12, min_weight_fraction_leaf = ' + str(
                                           min_weight_fraction_leaf), X, y, criterion='entropy',
                                       min_weight_fraction_leaf=min_weight_fraction_leaf, stratify=stratify,
                                       train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, GINI, min_samples_leaf=32, max_depth =12, min_weight_fraction_leaf = ' + str(
                                           min_weight_fraction_leaf), X, y, criterion='gini',
                                       min_weight_fraction_leaf=min_weight_fraction_leaf, stratify=stratify,
                                       train_size=train_size)

    # CALIBRATING max_features param + ENTROPY/GINI
    for max_features in range(1, 20):
        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 32, max_depth = 12, max_features = ' + str(
                                           max_features), X, y, criterion='entropy', max_features=max_features,
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics, 'DT, GINI, min_samples_leaf=32, max_depth =12, max_features = ' + str(
            max_features), X, y, criterion='gini', max_features=max_features, stratify=stratify, train_size=train_size)
    run_algorithm_dt_configuration(metrics, 'DT, ENTROPY, min_samples_leaf= 32, max_depth = 12, max_features = log2', X,
                                   y, criterion='entropy', max_features='log2', stratify=stratify,
                                   train_size=train_size)
    run_algorithm_dt_configuration(metrics, 'DT, GINI, min_samples_leaf=32, max_depth =12, max_features = log2', X, y,
                                   criterion='gini', max_features='log2', stratify=stratify, train_size=train_size)
    run_algorithm_dt_configuration(metrics, 'DT, ENTROPY, min_samples_leaf= 32, max_depth = 12, max_features = auto', X,
                                   y, criterion='entropy', max_features='auto', stratify=stratify,
                                   train_size=train_size)
    run_algorithm_dt_configuration(metrics, 'DT, GINI, min_samples_leaf=32, max_depth =12, max_features = auto', X, y,
                                   criterion='gini', max_features='auto', stratify=stratify, train_size=train_size)
    run_algorithm_dt_configuration(metrics, 'DT, ENTROPY, min_samples_leaf= 32, max_depth = 12, max_features = sqrt', X,
                                   y, criterion='entropy', max_features='sqrt', stratify=stratify,
                                   train_size=train_size)
    run_algorithm_dt_configuration(metrics, 'DT, GINI, min_samples_leaf=32, max_depth =12, max_features = sqrt', X, y,
                                   criterion='gini', max_features='sqrt', stratify=stratify, train_size=train_size)

    # CALIBRATING max_leaf_nodes param + ENTROPY/GINI
    for max_leaf_nodes in range(2, 100, 2):
        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 32, max_depth = 12, max_leaf_nodes = ' + str(
                                           max_leaf_nodes), X, y, criterion='entropy', max_leaf_nodes=max_leaf_nodes,
                                       stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics, 'DT, GINI, min_samples_leaf=32, max_depth =12, max_leaf_nodes = ' + str(
            max_leaf_nodes), X, y, criterion='gini', max_leaf_nodes=max_leaf_nodes, stratify=stratify,
                                       train_size=train_size)

    # CALIBRATING min_impurity_decrease param + ENTROPY/GINI
    for min_impurity_decrease in range(0, len(X), 20):
        run_algorithm_dt_configuration(metrics,
                                       'DT, ENTROPY, min_samples_leaf= 32, max_depth = 12, min_impurity_decrease = ' + str(
                                           min_impurity_decrease), X, y, criterion='entropy',
                                       min_impurity_decrease=min_impurity_decrease, stratify=stratify,
                                       train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, GINI, min_samples_leaf=32, max_depth =12, min_impurity_decrease = ' + str(
                                           min_impurity_decrease), X, y, criterion='gini',
                                       min_impurity_decrease=min_impurity_decrease, stratify=stratify,
                                       train_size=train_size)

    # CALIBRATING class_weight param + ENTROPY/GINI
    run_algorithm_dt_configuration(metrics,
                                   'DT, ENTROPY, min_samples_leaf= 32, max_depth = 12, class_weight = balanced', X, y,
                                   criterion='entropy', class_weight='balanced', stratify=stratify,
                                   train_size=train_size)
    run_algorithm_dt_configuration(metrics, 'DT, GINI, min_samples_leaf=32, max_depth =12, class_weight = balanced', X,
                                   y, criterion='gini', class_weight='balanced', stratify=stratify,
                                   train_size=train_size)

    # CALIBRATING ccp_alpha param + ENTROPY/GINI
    for ccp_alpha in [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1]:
        run_algorithm_dt_configuration(metrics, 'DT, ENTROPY, min_samples_leaf= 32, max_depth = 12, ccp_alpha = ' + str(
            ccp_alpha), X, y, criterion='entropy', ccp_alpha=ccp_alpha, stratify=stratify, train_size=train_size)
        run_algorithm_dt_configuration(metrics,
                                       'DT, GINI, min_samples_leaf=32, max_depth =12, ccp_alpha = ' + str(ccp_alpha), X,
                                       y, criterion='gini', ccp_alpha=ccp_alpha, stratify=stratify,
                                       train_size=train_size)
    #
    # TODO: our proposed DT more on it
    concatenated_min_samples_leaf = chain(range(10, 15), range(30, 35))
    #  10
    for min_samples_leaf in concatenated_min_samples_leaf:
        # 5 + 4 + 5 = 14
        concatenated_max_depth = chain(range(10, 15), range(20, 40, 5), range(85, 95, 2))
        for max_depth in concatenated_max_depth:
            # 7
            for max_leaf_nodes in range(20, 90, 10):
                concatenated_min_samples_split = chain(range(2, 10, 2), range(10, 100, 10), range(300, 320, 5))
                # 4+9+4=17
                for min_samples_split in concatenated_min_samples_split:
                    # 2.,
                    run_algorithm_dt_configuration(metrics, 'DT, ENTROPY, min_samples_leaf= '
                                                   + str(min_samples_leaf) + ', max_depth = ' + str(max_depth) +
                                                   ', max_leaf_nodes = ' + str(max_leaf_nodes) + ', min_samples_split = ' +
                                                   str(min_samples_split) + ', class_weight = balanced', X, y,
                                                   criterion='entropy',
                                                   min_samples_split=min_samples_split,
                                                   max_leaf_nodes=max_leaf_nodes, max_depth=max_depth,
                                                   min_samples_leaf=min_samples_leaf, class_weight='balanced',
                                                   stratify=stratify,
                                                   train_size=train_size)
                    run_algorithm_dt_configuration(metrics, 'DT, GINI, min_samples_leaf='
                                                   + str(min_samples_leaf) + ', max_depth = ' + str(max_depth) +
                                                   ', max_leaf_nodes = ' + str(max_leaf_nodes) + ', min_samples_split=' +
                                                   str(min_samples_split) + ', class_weight = balanced', X, y,
                                                   criterion='gini',
                                                   min_samples_split=min_samples_split,
                                                   max_leaf_nodes=max_leaf_nodes, max_depth=max_depth,
                                                   min_samples_leaf=min_samples_leaf, class_weight='balanced',
                                                   stratify=stratify,
                                                   train_size=train_size)
                metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_DT)

