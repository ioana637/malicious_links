import os
import sys
from datetime import datetime

from algorithms.ada import run_algorithm_ada_boost
from algorithms.gpc import run_algorithm_gpc
from algorithms.lr import run_algorithm_lr
from algorithms.mlp import run_algorithm_MLP
from algorithms.nb import run_algorithm_bnb, run_algorithm_gnb
from algorithms.qlda import run_algorithm_lda, run_algorithm_qda
from algorithms.svm import run_algorithm_SVC_linear_kernel, run_algorithm_SVC_sigmoid_kernel, \
    run_algorithm_SVC_RBF_kernel, run_algorithm_SVC_poly_kernel
from algorithms.xgb import run_algorithm_xgb

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M")


def main_mlp():
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.8]:
        run_algorithm_MLP('metrics_DN_standard_MLP_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                          stratify=True, train_size=train_size, normalize_data=True, scaler='standard')
        run_algorithm_MLP('metrics_DN_min_max_MLP_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                          stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')


def main_SVM_linear():
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.8]:
        run_algorithm_SVC_linear_kernel('metrics_DN_standard_SVC_linear_kernel_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                        stratify=True, train_size=train_size, normalize_data=True, scaler='standard')
        run_algorithm_SVC_linear_kernel('metrics_DN_min_max_SVC_linear_kernel_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                        stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')


def main_SVM_sigmoid():
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.8]:
        run_algorithm_SVC_sigmoid_kernel('metrics_DN_standard_SVM_sigmoid_kernel_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                         stratify=True, train_size=train_size, normalize_data=True, scaler='standard')
        run_algorithm_SVC_sigmoid_kernel('metrics_DN_min_max_SVM_sigmoid_kernel_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                         stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')


def main_SVM_RBF():
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.8]:
        run_algorithm_SVC_RBF_kernel('metrics_DN_standard_SVM_rbf_kernel_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                     stratify=True, train_size=train_size, normalize_data=True, scaler='standard')
        run_algorithm_SVC_RBF_kernel('metrics_DN_min_max_SVM_rbf_kernel_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                     stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')


def main_SVM_polynomial():
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.8]:
        run_algorithm_SVC_poly_kernel('metrics_DN_standard_SVM_polynomial_kernel_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                      stratify=True, train_size=train_size, normalize_data=True, scaler='standard')
        run_algorithm_SVC_poly_kernel('metrics_DN_min_max_SVM_polynomial_kernel_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                      stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')

def main_ada():
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.8]:
        run_algorithm_ada_boost('metrics_DN_standard_ADA_BOOST_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                stratify=True, train_size=train_size, normalize_data=True, scaler='standard')
        run_algorithm_ada_boost('metrics_DN_min_max_ADA_BOOST_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')


def main_lr():
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.8]:
        run_algorithm_lr('metrics_DN_standard_LR_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                         stratify=True, train_size=train_size, normalize_data=True, scaler='standard')
        run_algorithm_lr('metrics_DN_min_max_LR_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                         stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')


def main_nb():
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.8]:
        run_algorithm_gnb('metrics_DN_standard_GNB_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                          stratify=True, train_size=train_size, normalize_data=True, scaler='standard')
        run_algorithm_gnb('metrics_DN_min_max_GNB_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                          stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')
        run_algorithm_bnb('metrics_DN_standard_BNB_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                          stratify=True, train_size=train_size, normalize_data=True, scaler='standard')
        run_algorithm_bnb('metrics_DN_min_max_BNB_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                          stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')


def main_gpc():
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.8]:
        run_algorithm_gpc('metrics_DN_standard_GPC_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                          stratify=True, train_size=train_size, normalize_data=True, scaler='standard')
        run_algorithm_gpc('metrics_DN_min_max_GPC_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                          stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')


def main_qlda():
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.8]:
        # run_algorithm_qda('metrics_DN_standard_QDA_train_size_' + str(
        #     int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
        #                   stratify=True, train_size=train_size, normalize_data=True, scaler='standard')
        # run_algorithm_qda('metrics_DN_min_max_QDA_train_size_' + str(
        #     int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
        #                   stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')
        run_algorithm_lda('metrics_DN_standard_LDA_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                          stratify=True, train_size=train_size, normalize_data=True, scaler='standard')
        run_algorithm_lda('metrics_DN_min_max_LDA_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                          stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')

def main_xgb():
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.8]:
        run_algorithm_xgb('metrics_DN_standard_XGB_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                          stratify=True, train_size=train_size, normalize_data=True, scaler='standard')
        run_algorithm_xgb('metrics_DN_min_max_XGB_train_size_' + str(
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
    elif alg == 'lr':
        main_lr()
    elif alg == 'xgb':
        main_xgb()
    elif alg == 'qlda':
        main_qlda()
    elif alg == 'nb':
        main_nb()
    elif alg == 'gpc':
        main_gpc()
    elif alg == 'ada':
        main_ada()

# TODO test and run on server

main()
