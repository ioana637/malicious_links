import os

import numpy as np
from numpy import array
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from data_pre import split_data_in_testing_training, load_normalized_dataset
from utils import prediction, cal_metrics, appendMetricsTOCSV


def run_algorithm_ensemble_configuration(metrics, label, X, y, classifier1, classifier2, classifier3,
                                         train_size=0.8, stratify=False, no_classifiers = 3
                                         ):
    X_train, X_test, y_train, y_test = split_data_in_testing_training(X, y, stratify, train_size)

    # Performing training
    classifier1.fit(X_train, y_train)
    classifier2.fit(X_train, y_train)
    classifier3.fit(X_train, y_train)

    # Make predictions
    y_pred1, y_pred_probabilities1 = prediction(X_test, classifier1)
    y_pred2, y_pred_probabilities2 = prediction(X_test, classifier2)
    y_pred3, y_pred_probabilities3 = prediction(X_test, classifier3)

    lists_of_lists = [y_pred1, y_pred2, y_pred3]
    y_pred = list(map(lambda x:round(x/no_classifiers), [sum(x) for x in zip(*lists_of_lists)]))
    lists_of_lists = [y_pred_probabilities1.tolist(), y_pred_probabilities2.tolist(), y_pred_probabilities3.tolist()]
    # lists_of_lists.sum(axis=0)
    # # ceface= y_pred_probabilities1[:,1]
    # for i in range(0, len(y_pred)):
    #     print(y_pred_probabilities1[i][0], y_pred_probabilities1[i][1])
    #     print(y_pred_probabilities2[i][0], y_pred_probabilities2[i][1])
    #     print(y_pred_probabilities3[i][0], y_pred_probabilities3[i][1])

    sum_prob = [np.sum(x) for x in zip(*lists_of_lists)]
    y_pred_probabilities = array(map(lambda x:[round(x[0]/no_classifiers, 2), round(x[1]/no_classifiers, 2)],sum_prob))

    # Compute metrics
    precision, recall, f1, roc_auc = cal_metrics(y_test, y_pred, y_pred_probabilities, label, classifier3)
    metrics['label'].append(label)
    metrics['precision'].append(precision)
    metrics['recall'].append(recall)
    metrics['f1_score'].append(f1)
    metrics['roc_auc'].append(roc_auc)


def init_metrics_for_ensembles():
    return {'label': [],
            'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': []
            }


def run_algorithm_ensemble_dt_knn_rf(filename='', path='', stratify=True, train_size=0.8,
                                     normalize_data=True, scaler='min-max'):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_ensembles()

    my_filename = os.path.join(path, 'results\\ens', filename)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_ensembles, header=True)

    # Creating the classifier object
    # classifier1 = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
    #                                     splitter=splitter, min_samples_split=min_samples_split,
    #                                     min_weight_fraction_leaf=min_weight_fraction_leaf,
    #                                     max_features=max_features, max_leaf_nodes=max_leaf_nodes,
    #                                     min_impurity_decrease=min_impurity_decrease, ccp_alpha=ccp_alpha,
    #                                     class_weight='balanced'
    #                                     )
    classifier1 = DecisionTreeClassifier(class_weight='balanced')
    # classifier2 = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, p=p,
    #                                    leaf_size=leaf_size, metric=metric, metric_params=metric_params, n_jobs=-1)
    classifier2 = KNeighborsClassifier(n_jobs=-1)
    classifier3 = RandomForestClassifier(class_weight='balanced', n_jobs=-1)
    # classifier3 = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
    #                                      min_samples_leaf=min_samples_leaf,
    #                                      min_samples_split=min_samples_split, max_depth=max_depth,
    #                                      max_leaf_nodes=max_leaf_nodes, max_features=max_features,
    #                                      min_weight_fraction_leaf=min_weight_fraction_leaf,
    #                                      min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap,
    #                                      oob_score=oob_score, max_samples=max_samples, ccp_alpha=ccp_alpha,
    #                                      class_weight='balanced', n_jobs=-1)

    run_algorithm_ensemble_configuration(metrics, "KNN-DT-RF", X, y, classifier1, classifier2, classifier3,
                                         train_size=train_size, stratify=stratify)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_ensembles)

