import os
import sys
from datetime import datetime

from algorithms.ensembles import run_algorithm_ensemble_dt_knn_rf

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M")


def main_knn_dt_rf():
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.8]:
        run_algorithm_ensemble_dt_knn_rf(filename='metrics_DN_standard_KNN_DT_RF_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script, stratify=True,
                                         train_size=train_size, normalize_data=True, scaler='standard')
        # run_algorithm_ensemble_dt_knn_rf('metrics_DN_min_max_KNN_DT_RF_train_size_' + str(
        #     int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
        #                   stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')


def main():
    n = len(sys.argv)
    # print("\nName of Python script:", sys.argv[0])
    alg = sys.argv[1]
    if alg == 'knn-dt-rf':
        main_knn_dt_rf()


if __name__ == "__main__":
    main()
