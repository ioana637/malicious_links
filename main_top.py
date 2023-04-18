import os
from datetime import datetime

import pandas as pd

from algorithms.ada import run_best_configs_ada
from algorithms.dt import run_top_20_DT_configs
from algorithms.knn import run_top_20_KNN_configs
from algorithms.lr import run_best_configs_lr
from algorithms.nb import run_best_configs_gnb, run_best_configs_bnb
from algorithms.qlda import run_best_configs_lda
from algorithms.rfc import run_top_20_RFC_configs
from algorithms.svm import run_best_configs_SVM_linear, run_best_configs_SVM_poly, run_best_configs_SVM_sigmoid

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M")

# ----------------DECISION TREE----------------------------------------------------
path_to_script = os.path.dirname(os.path.abspath(__file__))
# run_top_20_DT_configs(filename = 'top20_DT_train_size_80_with_stratify_DN_minmax.csv', path = path_to_script, scaler='min-max', n_rep=100)
# run_top_20_DT_configs(filename = 'top20_DT_train_size_80_with_stratify_DN_standard.csv', path =path_to_script, scaler='standard', n_rep=100)

# ---------------------K-Nearest Neighbor ---------------------------------
# run_top_20_KNN_configs(filename = 'top20_KNN_train_size_80_with_stratify_DN_minmax.csv', path = path_to_script, scaler='min-max', n_rep=100)
# run_top_20_KNN_configs(filename = 'top20_KNN_train_size_80_with_stratify_DN_standard.csv', path = path_to_script, scaler='standard', n_rep=100)

# ------------------------RANDOM FOREST ------------------------------------------
# run_top_20_RFC_configs(filename = 'top20_RF_train_size_80_with_stratify_DN_minmax.csv', path = path_to_script, scaler='min-max', n_rep=100)
# run_top_20_RFC_configs(filename = 'top20_RF_train_size_80_with_stratify_DN_standard.csv', path = path_to_script, scaler='standard', n_rep=100)


# ------------------------------ADA BOOST-----------------------------------------
# df_configs_ada = pd.read_csv('new_results/ada/best.csv')
# run_best_configs_ada(df_configs_ada, filename='top20_ADA_train_size_80_with_stratify_DN_minmax.csv', path=path_to_script,
#                      stratify=True, train_size=0.8, normalize_data=True, scaler='min-max', n_rep=100)
# run_best_configs_ada(df_configs_ada, filename='top20_ADA_train_size_80_with_stratify_DN_standard.csv', path=path_to_script,
#                      stratify=True, train_size=0.8, normalize_data=True, scaler='standard', n_rep=100)


# ------------------------------Logistic Regression-----------------------------------------
# df_configs_lr = pd.read_csv('new_results/lr/best.csv')
# run_best_configs_lr(df_configs_lr, filename='top20_LR_train_size_80_with_stratify_DN_minmax.csv', path=path_to_script,
#                      stratify=True, train_size=0.8, normalize_data=True, scaler='min-max', n_rep=100)
# run_best_configs_lr(df_configs_lr, filename='top20_LR_train_size_80_with_stratify_DN_standard.csv', path=path_to_script,
#                      stratify=True, train_size=0.8, normalize_data=True, scaler='standard', n_rep=100)


# ------------------------------Naive Bayes -----------------------------------------
# df_configs_bnb = pd.read_csv('new_results/nb/best_bnb.csv')
# df_configs_gnb = pd.read_csv('new_results/nb/best_gnb.csv')
# run_best_configs_bnb(df_configs_bnb, filename='top20_BNB_train_size_80_with_stratify_DN_minmax.csv', path=path_to_script,
#                     stratify=True, train_size=0.8, normalize_data=True, scaler='min-max', n_rep=100)
# run_best_configs_bnb(df_configs_bnb, filename='top20_BNB_train_size_80_with_stratify_DN_standard.csv', path=path_to_script,
#                     stratify=True, train_size=0.8, normalize_data=True, scaler='standard', n_rep=100)
# run_best_configs_gnb(df_configs_gnb, filename='top20_GNB_train_size_80_with_stratify_DN_minmax.csv', path=path_to_script,
#                     stratify=True, train_size=0.8, normalize_data=True, scaler='min-max', n_rep=100)
# run_best_configs_gnb(df_configs_gnb, filename='top20_GNB_train_size_80_with_stratify_DN_standard.csv', path=path_to_script,
#                     stratify=True, train_size=0.8, normalize_data=True, scaler='standard', n_rep=100)


# ------------------------------ Linear Discriminant Analysis -----------------------------------------
df_configs_lda = pd.read_csv('new_results/qlda/best.csv')
run_best_configs_lda(df_configs_lda, filename='top20_LDA_train_size_80_with_stratify_DN_minmax.csv', path=path_to_script,
                    stratify=True, train_size=0.8, normalize_data=True, scaler='min-max', n_rep=100)
run_best_configs_lda(df_configs_lda, filename='top20_LDA_train_size_80_with_stratify_DN_standard.csv', path=path_to_script,
                    stratify=True, train_size=0.8, normalize_data=True, scaler='standard', n_rep=100)



# ----------------------------- SVM linear -----------------------------------------
# df_configs_svm_linear = pd.read_csv('new_results/svm/best_linear.csv')
# run_best_configs_SVM_linear(df_configs_svm_linear, filename='top20_SVM_linear_train_size_80_with_stratify_DN_minmax.csv', path=path_to_script,
#                      stratify=True, train_size=0.8, normalize_data=True, scaler='min-max', n_rep=100)
# run_best_configs_SVM_linear(df_configs_svm_linear, filename='top20_SVM_linear_train_size_80_with_stratify_DN_standard.csv', path=path_to_script,
#                      stratify=True, train_size=0.8, normalize_data=True, scaler='standard', n_rep=100)

#
# ------------------------------SVM polynomial -----------------------------------------
# df_configs_svm_poly = pd.read_csv('new_results/svm/best_poly.csv')
# run_best_configs_SVM_poly(df_configs_svm_poly, filename='top20_SVM_poly_train_size_80_with_stratify_DN_minmax.csv', path=path_to_script,
#                             stratify=True, train_size=0.8, normalize_data=True, scaler='min-max', n_rep=100)
# run_best_configs_SVM_poly(df_configs_svm_poly, filename='top20_SVM_poly_train_size_80_with_stratify_DN_standard.csv', path=path_to_script,
#                             stratify=True, train_size=0.8, normalize_data=True, scaler='standard', n_rep=100)
#
#
# ------------------------------SVM sigmoid -----------------------------------------
# df_configs_svm_sigmoid = pd.read_csv('new_results/svm/best_sigmoid.csv')
# run_best_configs_SVM_sigmoid(df_configs_svm_sigmoid, filename='top20_SVM_sigmoid_train_size_80_with_stratify_DN_minmax.csv', path=path_to_script,
#                           stratify=True, train_size=0.8, normalize_data=True, scaler='min-max', n_rep=100)
# run_best_configs_SVM_sigmoid(df_configs_svm_sigmoid, filename='top20_SVM_sigmoid_train_size_80_with_stratify_DN_standard.csv', path=path_to_script,
#                           stratify=True, train_size=0.8, normalize_data=True, scaler='standard', n_rep=100)
#