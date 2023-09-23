import os
from datetime import datetime

import pandas as pd

from algorithms.mlp import run_best_configs_mlp

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M")

# ----------------DECISION TREE----------------------------------------------------
path_to_script = os.path.dirname(os.path.abspath(__file__))


# df_configs_dt = pd.read_csv('new_results/dt/best-dt.csv')
# run_best_configs_DT(df_configs_dt, filename='top20_DT_train_size_80_with_stratify_DN_minmax.csv', path=path_to_script,
#                      stratify=True, train_size=0.8, normalize_data=True, scaler='min-max', n_rep=100)
# run_best_configs_DT(df_configs_dt, filename='top20_DT_train_size_80_with_stratify_DN_standard.csv', path=path_to_script,
#                      stratify=True, train_size=0.8, normalize_data=True, scaler='standard', n_rep=100)
# # # ---------------------K-Nearest Neighbor ---------------------------------
# df_configs_knn = pd.read_csv('new_results/knn/best-knn.csv')
# run_best_configs_knn(df_configs_knn, filename='top20_KNN_train_size_80_with_stratify_DN_minmax.csv', path=path_to_script,
#                      stratify=True, train_size=0.8, normalize_data=True, scaler='min-max', n_rep=100)
# run_best_configs_knn(df_configs_knn, filename='top20_KNN_train_size_80_with_stratify_DN_standard.csv', path=path_to_script,
#                      stratify=True, train_size=0.8, normalize_data=True, scaler='standard', n_rep=100)
# #
# # ------------------------RANDOM FOREST ------------------------------------------
# df_configs_rf = pd.read_csv('new_results/rfc/best-rfc.csv')
# run_best_configs_rfc(df_configs_rf, filename='top20_RFC_train_size_80_with_stratify_DN_minmax.csv', path=path_to_script,
#                     stratify=True, train_size=0.8, normalize_data=True, scaler='min-max', n_rep=100)
# run_best_configs_rfc(df_configs_rf, filename='top20_RFC_train_size_80_with_stratify_DN_standard.csv', path=path_to_script,
#                      stratify=True, train_size=0.8, normalize_data=True, scaler='standard', n_rep=100)


# ------------------------------ADA BOOST-----------------------------------------
# df_configs_ada = pd.read_csv('new_results/ada/best-ada.csv')
# run_best_configs_ada(df_configs_ada, filename='top20_ADA_train_size_80_with_stratify_DN_minmax.csv', path=path_to_script,
#                      stratify=True, train_size=0.8, normalize_data=True, scaler='min-max', n_rep=100)
# run_best_configs_ada(df_configs_ada, filename='top20_ADA_train_size_80_with_stratify_DN_standard.csv', path=path_to_script,
#                      stratify=True, train_size=0.8, normalize_data=True, scaler='standard', n_rep=100)
#
#
# # ------------------------------Logistic Regression-----------------------------------------
# df_configs_lr = pd.read_csv('new_results/lr/best-lr.csv')
# run_best_configs_lr(df_configs_lr, filename='top20_LR_train_size_80_with_stratify_DN_minmax.csv', path=path_to_script,
#                      stratify=True, train_size=0.8, normalize_data=True, scaler='min-max', n_rep=100)
# run_best_configs_lr(df_configs_lr, filename='top20_LR_train_size_80_with_stratify_DN_standard.csv', path=path_to_script,
#                      stratify=True, train_size=0.8, normalize_data=True, scaler='standard', n_rep=100)


# ------------------------------Naive Bayes -----------------------------------------
# df_configs_bnb = pd.read_csv('new_results/nb/best-bnb.csv')
# df_configs_gnb = pd.read_csv('new_results/nb/best-gnb.csv')
# run_best_configs_bnb(df_configs_bnb, filename='top20_BNB_train_size_80_with_stratify_DN_minmax.csv', path=path_to_script,
#                     stratify=True, train_size=0.8, normalize_data=True, scaler='min-max', n_rep=100)
# run_best_configs_bnb(df_configs_bnb, filename='top20_BNB_train_size_80_with_stratify_DN_standard.csv', path=path_to_script,
#                     stratify=True, train_size=0.8, normalize_data=True, scaler='standard', n_rep=100)
# run_best_configs_gnb(df_configs_gnb, filename='top20_GNB_train_size_80_with_stratify_DN_minmax.csv', path=path_to_script,
#                     stratify=True, train_size=0.8, normalize_data=True, scaler='min-max', n_rep=100)
# run_best_configs_gnb(df_configs_gnb, filename='top20_GNB_train_size_80_with_stratify_DN_standard.csv', path=path_to_script,
#                     stratify=True, train_size=0.8, normalize_data=True, scaler='standard', n_rep=100)


# ------------------------------ Linear Discriminant Analysis -----------------------------------------
# df_configs_lda = pd.read_csv('new_results/qlda/best-lda.csv')
# run_best_configs_lda(df_configs_lda, filename='top20_LDA_train_size_80_with_stratify_DN_minmax.csv', path=path_to_script,
#                     stratify=True, train_size=0.8, normalize_data=True, scaler='min-max', n_rep=100)
# run_best_configs_lda(df_configs_lda, filename='top20_LDA_train_size_80_with_stratify_DN_standard.csv', path=path_to_script,
#                     stratify=True, train_size=0.8, normalize_data=True, scaler='standard', n_rep=100)



# ----------------------------- SVM linear -----------------------------------------
# df_configs_svm_linear = pd.read_csv('new_results/svc/best-linear.csv')
# run_best_configs_SVM_linear(df_configs_svm_linear, filename='top20_SVM_linear_train_size_80_with_stratify_DN_minmax.csv', path=path_to_script,
#                      stratify=True, train_size=0.8, normalize_data=True, scaler='min-max', n_rep=100)
# run_best_configs_SVM_linear(df_configs_svm_linear, filename='top20_SVM_linear_train_size_80_with_stratify_DN_standard.csv', path=path_to_script,
#                      stratify=True, train_size=0.8, normalize_data=True, scaler='standard', n_rep=100)

#
# ------------------------------SVM polynomial -----------------------------------------
# df_configs_svm_poly = pd.read_csv('new_results/svc/best-poly.csv')
# run_best_configs_SVM_poly(df_configs_svm_poly, filename='top20_SVM_poly_train_size_80_with_stratify_DN_minmax.csv', path=path_to_script,
#                             stratify=True, train_size=0.8, normalize_data=True, scaler='min-max', n_rep=100)
# run_best_configs_SVM_poly(df_configs_svm_poly, filename='top20_SVM_poly_train_size_80_with_stratify_DN_standard.csv', path=path_to_script,
#                             stratify=True, train_size=0.8, normalize_data=True, scaler='standard', n_rep=100)

#
# ------------------------------SVM sigmoid -----------------------------------------
# df_configs_svm_sigmoid = pd.read_csv('new_results/svc/best-sigmoid.csv')
# run_best_configs_SVM_sigmoid(df_configs_svm_sigmoid, filename='top20_SVM_sigmoid_train_size_80_with_stratify_DN_minmax.csv', path=path_to_script,
#                           stratify=True, train_size=0.8, normalize_data=True, scaler='min-max', n_rep=100)
# run_best_configs_SVM_sigmoid(df_configs_svm_sigmoid, filename='top20_SVM_sigmoid_train_size_80_with_stratify_DN_standard.csv', path=path_to_script,
#                           stratify=True, train_size=0.8, normalize_data=True, scaler='standard', n_rep=100)
#
# ------------------------------SVM RBF kernel -----------------------------------------
# df_configs_svm_RBF = pd.read_csv('new_results/svc/best-rbf.csv')
# run_best_configs_SVM_RBF(df_configs_svm_RBF, filename='top20_SVM_RBF_train_size_80_with_stratify_DN_minmax.csv', path=path_to_script,
#                           stratify=True, train_size=0.8, normalize_data=True, scaler='min-max', n_rep=100)
# run_best_configs_SVM_RBF(df_configs_svm_RBF, filename='top20_SVM_RBF_train_size_80_with_stratify_DN_standard.csv', path=path_to_script,
#                           stratify=True, train_size=0.8, normalize_data=True, scaler='standard', n_rep=100)


# ----------------------------------------XGB----------------------------------------
# df_configs_xgb = pd.read_csv('new_results/xgb/best-xgb.csv')
# run_best_configs_xgb(df_configs_xgb, filename='top20_XGB_train_size_80_with_stratify_DN_minmax.csv', path=path_to_script,
#                          stratify=True, train_size=0.8, normalize_data=True, scaler='min-max', n_rep=100)
# run_best_configs_xgb(df_configs_xgb, filename='top20_XGB_train_size_80_with_stratify_DN_standard.csv', path=path_to_script,
#                          stratify=True, train_size=0.8, normalize_data=True, scaler='standard', n_rep=100)


# ----------------------------------------MLP----------------------------------------
df_configs_mlp = pd.read_csv('../new_results/mlp/best-mlp-1.csv')
run_best_configs_mlp(df_configs_mlp, filename='top20_MLP_train_size_80_with_stratify_DN_minmax.csv', path=path_to_script,
                         stratify=True, train_size=0.8, normalize_data=True, scaler='min-max', n_rep=100)
run_best_configs_mlp(df_configs_mlp, filename='top20_MLP_train_size_80_with_stratify_DN_standard.csv', path=path_to_script,
                         stratify=True, train_size=0.8, normalize_data=True, scaler='standard', n_rep=100)


# ----------------------------------------GPC----------------------------------------
# df_configs_gpc = pd.read_csv('new_results/gpc/best-gpc.csv')
# run_best_configs_GPC(df_configs_gpc, filename='top20_GPC_train_size_80_with_stratify_DN_minmax.csv', path=path_to_script,
#                      stratify=True, train_size=0.8, normalize_data=True, scaler='min-max', n_rep=100)
# run_best_configs_GPC(df_configs_gpc, filename='top20_GPC_train_size_80_with_stratify_DN_standard.csv', path=path_to_script,
#                      stratify=True, train_size=0.8, normalize_data=True, scaler='standard', n_rep=100)