import os
from datetime import datetime

from algorithms.dt import run_algorithm_dt
from algorithms.knn import run_algorithm_KNN
from algorithms.rfc import run_algorithm_rf

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M")


def main_rfc():
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        run_algorithm_rf('metrics_DN_min_max_RFC_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                         stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')
        run_algorithm_rf('metrics_DN_standard_RFC_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                         stratify=True, train_size=train_size, normalize_data=True, scaler='standard')


def main_dt():
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        run_algorithm_dt('metrics_DN_min_max_DT_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                         stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')
        run_algorithm_dt('metrics_DN_standard_DT_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                         stratify=True, train_size=train_size, normalize_data=True, scaler='standard')

def main_knn():
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        run_algorithm_KNN('metrics_DN_min_max_KNN_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                         stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')
        run_algorithm_KNN('metrics_DN_standard_KNN_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                         stratify=True, train_size=train_size, normalize_data=True, scaler='standard')


main_knn()
main_dt()
main_rfc()
