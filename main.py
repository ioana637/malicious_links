import os
from datetime import datetime

import sklearn

from algorithms.dt import run_algorithm_dt, run_algorithm_dt_with_k_fold
from algorithms.knn import run_algorithm_KNN, run_algorithm_KNN_with_k_fold
from algorithms.rfc import run_algorithm_rf, run_algorithm_rf_with_k_fold

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M")


def main_rfc():
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    run_algorithm_rf('metrics_DN_standard_RFC_train_size_' + str(
        int(0.4 * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                     stratify=True, train_size=0.4, normalize_data=True, scaler='standard')

    for train_size in [ 0.5, 0.6]:
        run_algorithm_rf('metrics_DN_min_max_RFC_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                         stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')
        run_algorithm_rf('metrics_DN_standard_RFC_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                         stratify=True, train_size=train_size, normalize_data=True, scaler='standard')


def main_dt():
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.8]:
        run_algorithm_dt('metrics_DN_min_max_DT_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                         stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')
        run_algorithm_dt('metrics_DN_standard_DT_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                         stratify=True, train_size=train_size, normalize_data=True, scaler='standard')

def main_knn():
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.8]:
        run_algorithm_KNN('metrics_DN_min_max_KNN_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                         stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')
        run_algorithm_KNN('metrics_DN_standard_KNN_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                         stratify=True, train_size=train_size, normalize_data=True, scaler='standard')


# main_knn()
# main_dt()
# main_rfc()

if __name__ == '__main__':
    print(sorted(sklearn.metrics.SCORERS.keys()))
    print('DT------------------')
    run_algorithm_dt_with_k_fold(scaler='standard')
    print('KNN------------------')
    run_algorithm_KNN_with_k_fold(scaler='min-max')
    print('RF------------------')
    run_algorithm_rf_with_k_fold(scaler='min-max')

    # KNN, weights = distance,  metric = euclidean, p=5, algorithm = ball_tree, n_neighbors = 3, leaf size= 86; scaler min max
# 86.60158656	80.43086931	96.02540471	0.92785271

# RF, entropy, n_estimators = 110, max_depth=53, min_samples_leaf=3, max_leaf_nodes=1564, min_samples_split=4,
# class_weight=balanced
# 87.60161562	82.55175553	96.39817176	98.08577135
# scaler - min max
