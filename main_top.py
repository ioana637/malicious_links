import os
from datetime import datetime

import pandas as pd

from algorithms.dt import run_top_20_DT_configs
from algorithms.knn import run_top_20_KNN_configs
from algorithms.rfc import run_top_20_RFC_configs

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
run_top_20_RFC_configs(filename = 'top20_RF_train_size_80_with_stratify_DN_minmax.csv', path = path_to_script, scaler='min-max', n_rep=100)
run_top_20_RFC_configs(filename = 'top20_RF_train_size_80_with_stratify_DN_standard.csv', path = path_to_script, scaler='standard', n_rep=100)
