import os
import sys
from datetime import datetime
import sklearn

from algorithms.ada import run_algorithm_ada_parallel
from algorithms.dt import run_algorithm_dt_parallel
from algorithms.gpc import run_algorithm_gpc_parallel
from algorithms.knn import run_algorithm_KNN_parallel
from algorithms.lr import run_algorithm_lr_parallel
from algorithms.mlp import run_algorithm_MLP_parallel
from algorithms.nb import run_algorithm_bnb_parallel, run_algorithm_gnb_parallel
from algorithms.qlda import run_algorithm_lda_parallel
from algorithms.rfc import run_algorithm_rfc_parallel
from algorithms.svm import run_algorithm_SVC_linear_kernel_parallel, \
    run_algorithm_SVC_RBF_kernel_parallel, run_algorithm_SVC_sigmoid_kernel_parallel, \
    run_algorithm_SVC_poly_kernel_parallel
from algorithms.xgb import run_algorithm_xgb_parallel

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M")

def parallel_main_mlp(no_threads):
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.8]:
        run_algorithm_MLP_parallel('metrics_DN_min_max_MLP_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                          stratify=True, train_size=train_size, normalize_data=True, scaler='min-max')
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


def parallel_main_SVM_sigmoid(no_threads):
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.8]:
        run_algorithm_SVC_sigmoid_kernel_parallel('metrics_DN_standard_SVM_sigmoid_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                   stratify=True, train_size=train_size, normalize_data=True, scaler='standard',
                                   no_threads=no_threads)
        run_algorithm_SVC_sigmoid_kernel_parallel('metrics_DN_min_max_SVM_sigmoid_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                   stratify=True, train_size=train_size, normalize_data=True, scaler='min-max',
                                   no_threads=no_threads)


def parallel_main_SVM_poly(no_threads):
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.8]:
        run_algorithm_SVC_poly_kernel_parallel('metrics_DN_standard_SVM_poly_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                                  stratify=True, train_size=train_size, normalize_data=True, scaler='standard',
                                                  no_threads=no_threads)
        run_algorithm_SVC_poly_kernel_parallel('metrics_DN_min_max_SVM_poly_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                                  stratify=True, train_size=train_size, normalize_data=True, scaler='min-max',
                                                  no_threads=no_threads)


def parallel_main_ADA(no_threads):
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.8]:
        run_algorithm_ada_parallel('metrics_DN_standard_ADA_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                               stratify=True, train_size=train_size, normalize_data=True, scaler='standard',
                                               no_threads=no_threads)
        run_algorithm_ada_parallel('metrics_DN_min_max_ADA_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                               stratify=True, train_size=train_size, normalize_data=True, scaler='min-max',
                                               no_threads=no_threads)

def parallel_main_lr(no_threads):
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.8]:
        run_algorithm_lr_parallel('metrics_DN_standard_LR_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                   stratify=True, train_size=train_size, normalize_data=True, scaler='standard',
                                   no_threads=no_threads)
        run_algorithm_lr_parallel('metrics_DN_min_max_LR_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                   stratify=True, train_size=train_size, normalize_data=True, scaler='min-max',
                                   no_threads=no_threads)

def parallel_main_BNB(no_threads):
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.8]:
        run_algorithm_bnb_parallel('metrics_DN_standard_BNB_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                  stratify=True, train_size=train_size, normalize_data=True, scaler='standard',
                                  no_threads=no_threads)
        run_algorithm_bnb_parallel('metrics_DN_min_max_BNB_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                  stratify=True, train_size=train_size, normalize_data=True, scaler='min-max',
                                  no_threads=no_threads)


def parallel_main_GNB(no_threads):
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.8]:
        run_algorithm_gnb_parallel('metrics_DN_standard_GNB_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                  stratify=True, train_size=train_size, normalize_data=True, scaler='standard',
                                  no_threads=no_threads)
        run_algorithm_gnb_parallel('metrics_DN_min_max_GNB_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                  stratify=True, train_size=train_size, normalize_data=True, scaler='min-max',
                                  no_threads=no_threads)


def parallel_main_LDA(no_threads):
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.8]:
        run_algorithm_lda_parallel('metrics_DN_standard_LDA_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                  stratify=True, train_size=train_size, normalize_data=True, scaler='standard',
                                  no_threads=no_threads)
        run_algorithm_lda_parallel('metrics_DN_min_max_LDA_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                  stratify=True, train_size=train_size, normalize_data=True, scaler='min-max',
                                  no_threads=no_threads)


def parallel_main_knn(no_threads):
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.8]:
        run_algorithm_KNN_parallel('metrics_DN_standard_KNN_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                   stratify=True, train_size=train_size, normalize_data=True, scaler='standard',
                                   no_threads=no_threads)
        run_algorithm_KNN_parallel('metrics_DN_min_max_KNN_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                   stratify=True, train_size=train_size, normalize_data=True, scaler='min-max',
                                   no_threads=no_threads)


def parallel_main_DT(no_threads):
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.8]:
        run_algorithm_dt_parallel('metrics_DN_standard_DT_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                   stratify=True, train_size=train_size, normalize_data=True, scaler='standard',
                                   no_threads=no_threads)
        run_algorithm_dt_parallel('metrics_DN_min_max_DT_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                   stratify=True, train_size=train_size, normalize_data=True, scaler='min-max',
                                   no_threads=no_threads)


def parallel_main_rfc(no_threads):
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    for train_size in [0.8]:
        run_algorithm_rfc_parallel('metrics_DN_standard_RF_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                   stratify=True, train_size=train_size, normalize_data=True, scaler='standard',
                                   no_threads=no_threads)
        run_algorithm_rfc_parallel('metrics_DN_min_max_RF_train_size_' + str(
            int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv', path=path_to_script,
                                   stratify=True, train_size=train_size, normalize_data=True, scaler='min-max',
                                   no_threads=no_threads)


def main():
    n = len(sys.argv)
    # print("\nName of Python script:", sys.argv[0])
    alg = sys.argv[1]
    threads = int(sys.argv[2]) # no. of threads created for running
    if alg == 'mlp':
        parallel_main_mlp(threads) # run last
    elif alg == 'gpc':
        parallel_main_gpc(threads) # run last
    elif alg == 'svc-linear':
        parallel_main_SVM_linear(threads) #run
    elif alg == 'svc-rbf':
        parallel_main_SVM_RBF(threads) #run
    elif alg == 'xgb':
        parallel_main_xgb(threads) #run with error
    elif alg == 'svc-sigmoid':
        parallel_main_SVM_sigmoid(threads) #run
    elif alg == 'svc-poly':
        parallel_main_SVM_poly(threads) #running
    elif alg == 'ada':
        parallel_main_ADA(threads) #running todo
    elif alg == 'lr':
        parallel_main_lr(threads) #run
    elif alg == 'bnb':
        parallel_main_BNB(threads) #run
    elif alg == 'gnb':
        parallel_main_GNB(threads) #run
    elif alg == 'lda':
        parallel_main_LDA(threads) #run
    elif alg == 'knn':
        parallel_main_knn(threads) #running
    elif alg == 'rf':
        parallel_main_rfc(threads) #running
    elif alg == 'dt':
        parallel_main_DT(threads) #running



if __name__ == "__main__":
    main()


