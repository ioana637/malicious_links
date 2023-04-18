import os
import sys
from datetime import datetime
import sklearn

from algorithms.gpc import run_algorithm_gpc_parallel
from algorithms.mlp import run_algorithm_MLP_parallel
from algorithms.svm import run_algorithm_SVC_linear_kernel_parallel, \
    run_algorithm_SVC_RBF_kernel_parallel
from algorithms.xgb import run_algorithm_xgb_parallel

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M")

def parallel_main_mlp(no_threads):
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.8]:
        # run_algorithm_MLP_parallel('metrics_DN_min_max_MLP_train_size_' + str(
        #     int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
        #                   stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')
        run_algorithm_MLP_parallel('metrics_DN_standard_MLP_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                   stratify=True, train_size=train_size, normalize_data=True, scaler='standard',
                                   no_threads=no_threads)


def parallel_main_SVM_linear(no_threads):
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.8]:
        run_algorithm_SVC_linear_kernel_parallel('metrics_DN_standard_SVC_linear_kernel_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                        stratify=True, train_size=train_size, normalize_data=True, scaler='standard',
                                                 no_threads=no_threads)
        run_algorithm_SVC_linear_kernel_parallel('metrics_DN_min_max_SVC_linear_kernel_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                        stratify=True, train_size=train_size, normalize_data=True, scaler='min-max',
                                                 no_threads=no_threads)

def parallel_main_SVM_RBF(no_threads):
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.8]:
        run_algorithm_SVC_RBF_kernel_parallel('metrics_DN_standard_SVM_rbf_kernel_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                     stratify=True, train_size=train_size, normalize_data=True, scaler='standard',
                                     no_threads=no_threads)
        run_algorithm_SVC_RBF_kernel_parallel('metrics_DN_min_max_SVM_rbf_kernel_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                     stratify=True, train_size=train_size, normalize_data=True, scaler='min-max',
                                     no_threads=no_threads)

def parallel_main_xgb(no_threads):
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.8]:
        run_algorithm_xgb_parallel('metrics_DN_standard_XGB_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                          stratify=True, train_size=train_size, normalize_data=True, scaler='standard',
                          no_threads=no_threads)
        run_algorithm_xgb_parallel('metrics_DN_min_max_XGB_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                          stratify=True, train_size=train_size, normalize_data=True, scaler='min-max',
                          no_threads=no_threads)


def parallel_main_gpc(no_threads):
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.8]:
        run_algorithm_gpc_parallel('metrics_DN_standard_GPC_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                          stratify=True, train_size=train_size, normalize_data=True, scaler='standard',
                          no_threads=no_threads)
        run_algorithm_gpc_parallel('metrics_DN_min_max_GPC_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                          stratify=True, train_size=train_size, normalize_data=True, scaler='min-max',
                          no_threads=no_threads)


def main():
    n = len(sys.argv)
    # print("\nName of Python script:", sys.argv[0])
    alg = sys.argv[1]
    threads = int(sys.argv[2]) # no. of threads created for running
    if alg == 'mlp':
        parallel_main_mlp(threads)
    elif alg == 'svc-linear':
        parallel_main_SVM_linear(threads)
    elif alg == 'svc-rbf':
        parallel_main_SVM_RBF(threads)
    elif alg == 'xgb':
        parallel_main_xgb(threads)
    elif alg == 'gpc':
        parallel_main_gpc(threads)


if __name__ == "__main__":
    main()


