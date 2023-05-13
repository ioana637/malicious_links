import os
import sys
from datetime import datetime
from itertools import combinations

import pandas as pd

from algorithms.ensembles import run_algorithm_ensemble, init_metrics_for_ensembles
from algorithms.enums import Algorithms
from data_post import compute_average_metric, compute_roc_auc_score_100
from utils import appendMetricsTOCSV

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M")


def run_ensemble_algs_parallel(algs: [Algorithms], no_repeats=10, train_size=0.8, filename_standard='',
                               filename_min_max=''):
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    run_algorithm_ensemble(filename=filename_min_max, path=path_to_script, stratify=True, train_size=train_size,
                           normalize_data=True, scaler='min-max', no_repeats=no_repeats, ens=algs)
    run_algorithm_ensemble(filename=filename_standard, path=path_to_script, stratify=True, train_size=train_size,
                           normalize_data=True, scaler='standard', no_repeats=no_repeats, ens=algs)


def main_parallel():
    n = len(sys.argv)
    # print("\nName of Python script:", sys.argv[0])
    # algs = [alg for alg in Algorithms]
    algs = [Algorithms.ADA, Algorithms.LR, Algorithms.BNB, Algorithms.LDA,
            Algorithms.XGB, Algorithms.KNN, Algorithms.DT, Algorithms.RF, Algorithms.SVM_rbf]

    # algs = [Algorithms.XGB, Algorithms.RF, Algorithms.SVM_rbf]
    combos = list(combinations(algs, 3))

    filename_min_max = init_ens_files(train_size=0.8, scaler='min-max')
    filename_standard = init_ens_files(train_size=0.8, scaler='standard')

    for combo in combos:
        run_ensemble_algs_parallel(combo, no_repeats=100, train_size=0.8, filename_standard=filename_standard,
                                   filename_min_max=filename_min_max)


def init_ens_files(train_size=0.8, scaler='min-max'):
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    filename = 'metrics_DN_' + scaler + '_ENSEMBLE_train_size_' + str(
        int(train_size * 100)) + '_with_stratify_' + dt_string + '.csv'
    my_filename = os.path.join(path_to_script, 'new_results\\ens', filename)
    metrics = init_metrics_for_ensembles()
    metrics_df = pd.DataFrame(metrics)
    metrics_df = compute_average_metric(metrics_df)
    metrics_df = compute_roc_auc_score_100(metrics_df)
    metrics = appendMetricsTOCSV(my_filename, metrics_df, init_metrics_for_ensembles, header=True)
    return filename


if __name__ == "__main__":
    main_parallel()
