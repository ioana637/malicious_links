# Analyze results files
# 1. Load data from result files
# 2. Compute average metric for each row (avg(precision, recall, F1_score, ROC_AUC*100))
# 3. Overall average, min, max
# 4. For 70 and 80 train size - for each paramater - average (sort parameters values asc)
# 5. List top 20 configurations based on the average_metric, sort their parameters
import json
import os

import pandas as pd

# 1. Load data from result files
from utils import unique, print_dict


def load_results_from_file(filename):
    try:
        df = pd.read_csv(filename)
        return df
    except FileNotFoundError as err:
        print(err)

# 2. Compute average metric for each row (avg(precision, recall, F1_score, ROC_AUC*100))
def compute_average_metric(df):
    # metrics names: precision,recall,f1_score,roc_auc
    df = df.assign(average_metric=lambda x: (x.precision + x.recall + x.f1_score + x.roc_auc *100)/4)
    return df

# 3. Overall average, min, max for precision, recall, F1 score, ROC_AUC, average_metric
def compute_average_for_metrics(df):
    # avg_precision, avg_recall, avg_F1_score, avg_ROC_AUC, avg_average_metric
    return df[["precision","recall", "f1_score", "roc_auc", "average_metric"]].mean()

def compute_min_for_metrics(df):
    # min_precision, min_recall, min_F1_score, min_ROC_AUC, min_average_metric
    return df[["precision","recall", "f1_score", "roc_auc", "average_metric"]].min()

def compute_max_for_metrics(df):
    # max_precision, max_recall, max_F1_score, max_ROC_AUC, max_average_metric
    return df[["precision","recall", "f1_score", "roc_auc", "average_metric"]].max()

# 4. For 70 and 80 train size - for each paramater - average (sort parameters values asc)
def compute_avg_for_each_parameter(df):
    for (columnName, columnData) in df.iteritems():
        if (columnName in ['label','precision', 'recall', 'f1_score', 'roc_auc','average_metric' ]):
            continue
        try:
            df_aux = df.groupby(columnName)[['precision', 'recall', 'f1_score', 'roc_auc','average_metric']].mean()
            print()
            print(df_aux.to_string())
        except FutureWarning as err:
            print(columnName)
        # print('Column Name : ', columnName)
        # print('Column Contents : ', columnData.values)

# 5. List top 20 configurations based on the average_metric, sort their parameters
def determine_top_configuration(df, top_k = 20):
    df.sort_values(by =['average_metric'], ascending=False, inplace=True)
    top_k_df = df.head(top_k)
    # print parameters of the algorithm
    # all columns, excepting label, precision, recall, f1_score, roc_auc,average_metric
    params = {}
    for (columnName, columnData) in top_k_df.iteritems():
        if (columnName in ['label','precision', 'recall', 'f1_score', 'roc_auc','average_metric' ]):
            continue
        unique_values = unique(columnData.values)
        # print(unique_values)
        params[columnName] = unique_values
        print('Column Name : ', columnName)
        print('Column Contents : ', unique_values)
    return top_k_df, params

def main_data_post():
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    # my_filename = os.path.join(path_to_script, 'new_results/knn', filename) # for Linux
    my_filename = os.path.join(path_to_script, 'new_results\\knn', 'metrics_DN_min_max_KNN_train_size_20_with_stratify_04_02_2023_08_06.csv')

    df = load_results_from_file(my_filename)
    df = compute_average_metric(df)
    print('--------------------- AVERAGE ---------------------')
    print(compute_average_for_metrics(df))
    print('--------------------- MAX ---------------------')
    print(compute_max_for_metrics(df))
    print('--------------------- MIN ---------------------')
    print(compute_min_for_metrics(df))

    top_k_df, params = determine_top_configuration(df)
    print(top_k_df.to_string())
    print_dict(params)

    compute_avg_for_each_parameter(df)

main_data_post()