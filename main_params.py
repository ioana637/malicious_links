import os
import sys
from datetime import datetime

from algorithms.mlp import run_algorithm_MLP
from algorithms.svm import run_algorithm_SVC_linear_kernel, run_algorithm_SVC_sigmoid_kernel, \
    run_algorithm_SVC_RBF_kernel, run_algorithm_SVC_poly_kernel

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M")


def main_mlp():
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.8]:
        run_algorithm_MLP('metrics_DN_standard_MLP_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                          stratify=True, train_size=train_size, normalize_data=True, scaler='standard')
        run_algorithm_MLP('metrics_DN_standard_MLP_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                          stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')


def main_SVM_linear():
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.8]:
        run_algorithm_SVC_linear_kernel('metrics_DN_standard_SVC_linear_kernel_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                        stratify=True, train_size=train_size, normalize_data=True, scaler='standard')
        run_algorithm_SVC_linear_kernel('metrics_DN_standard_SVC_linear_kernel_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                        stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')


def main_SVM_sigmoid():
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.8]:
        run_algorithm_SVC_sigmoid_kernel('metrics_DN_standard_SVM_sigmoid_kernel_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                         stratify=True, train_size=train_size, normalize_data=True, scaler='standard')
        run_algorithm_SVC_sigmoid_kernel('metrics_DN_standard_SVM_sigmoid_kernel_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                         stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')


def main_SVM_RBF():
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.8]:
        run_algorithm_SVC_RBF_kernel('metrics_DN_standard_SVM_rbf_kernel_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                     stratify=True, train_size=train_size, normalize_data=True, scaler='standard')
        run_algorithm_SVC_RBF_kernel('metrics_DN_standard_SVM_rbf_kernel_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                     stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')


def main_SVM_polynomial():
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.8]:
        run_algorithm_SVC_poly_kernel('metrics_DN_standard_SVM_polynomial_kernel_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                      stratify=True, train_size=train_size, normalize_data=True, scaler='standard')
        run_algorithm_SVC_poly_kernel('metrics_DN_standard_SVM_polynomial_kernel_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                      stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')


def main():
    n = len(sys.argv)
    # print("\nName of Python script:", sys.argv[0])
    alg = sys.argv[1]
    if alg == 'mlp':
        main_mlp()
    elif alg == 'svc-linear':
        main_SVM_linear()
    elif alg == 'svc-rbf':
        main_SVM_RBF()
    elif alg == 'svc-sigmoid':
        main_SVM_sigmoid()
    elif alg == 'svc-poly':
        main_SVM_polynomial()

# TODO test and run on server

main()
