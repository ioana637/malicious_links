import os
import pandas as pd

from dt import init_metrics_for_DT, run_algorithm_dt_configuration_feature_importance
from rfc import run_algorithm_rfc_configuration_feature_importance, init_metrics_for_rfc

inputFile = r'data\output_1.csv'
cwd = os.getcwd()
df = pd.read_csv(f'{cwd}\\{inputFile}')
df["features"] = df["features"].str.replace('[', '', regex = True)
df["features"] = df["features"].str.replace(']', '', regex = True)
feature_names = ['index', 'url_length',
                 'number_special_characters', 'charset',
                 'server', 'content_length', 'whois_country',
                 'whois_statepro', 'whois_regdate',
                 'whois_updated_date', 'tcp_conversation_exchange',
                 'dist_remote_tcp_port', 'remote_ips', 'app_bytes',
                 'source_app_packets', 'remote_app_packets',
                 'source_app_bytes', 'remote_app_bytes',
                 'app_packets', 'dns_query_times']
df[feature_names] = df["features"].str.split(',', expand = True)
df = df.drop('features', axis=1)
df = df.drop('index', axis=1)
feature_names.remove('index')

count_class_balance = df.apply(lambda x : True if x['label'] == 0 else False, axis = 1)
#   # 0 - malicious; 1 - benign
#   # False for benign and True for malicious
# # Count number of True in the series
num_rows_malicious = len(count_class_balance[count_class_balance == True].index)
print('num_rows_malicious', num_rows_malicious)
num_rows_benign = len(count_class_balance[count_class_balance == False].index)
print('num_rows_benign', num_rows_benign)

df_preprocessed = df.copy(deep=True)
df_normalized = df.copy(deep=True)

y = df['label'].copy()
X = df.drop('label', axis=1).copy()
metrics = init_metrics_for_rfc()


run_algorithm_rfc_configuration_feature_importance(metrics,
                                'RF, GINI, n_estimators = 100, max_depth=12, min_samples_leaf=3, max_leaf_nodes=1336, min_samples_split=4, class_weight=balanced',
                                X, y, criterion = 'gini', n_estimators= 100,
                                min_samples_leaf = 3, max_depth=12, min_samples_split= 4, max_leaf_nodes=1336,
                                                   class_weight='balanced',
                                stratify = True, train_size = 0.8, normalize_data = True, scaler = 'standard',
                                                   feature_names=feature_names)

# run_algorithm_rfc_configuration_feature_importance(metrics,
#                                 'RF, ENTROPY, n_estimators = 100, min_samples_leaf=1, max_depth = 12, min_samples_split = 5, max_leaf_nodes = 1304',
#                                 X, y, criterion = 'entropy', n_estimators= 100,
#                                 min_samples_leaf = 1, max_depth=12, min_samples_split= 5, max_leaf_nodes=1304,
#                                 stratify = True, train_size = 0.8, normalize_data = True, scaler = 'standard',
#                                                    feature_names=feature_names)
metrics = init_metrics_for_DT()
run_algorithm_dt_configuration_feature_importance(metrics,
                                                  'DT, ENTROPY, min_samples_leaf= 11, max_depth = 20, max_leaf_nodes = 40, min_samples_split = 20, class_weight = balanced',
                                                  X, y, criterion = 'entropy', max_features = 'sqrt', max_depth=20, max_leaf_nodes=40,
                                                  min_samples_split=20, stratify = True, min_samples_leaf = 11,
                                                  train_size=0.8, normalize_data = True, scaler = 'min-max',
                                                  class_weight = 'balanced',
                                                  feature_names=feature_names)

# run_algorithm_dt_configuration_feature_importance(metrics,
#                                                   'DT, ENTROPY, min_samples_leaf=11, max_depth =93, max_leaf_nodes = 25, min_samples_split = 22',
#                                                   X, y, criterion = 'entropy', max_features = 'sqrt', max_depth=93, max_leaf_nodes=25,
#                                                   min_samples_split=22, stratify = True,
#                                                   train_size=0.8, normalize_data = True, scaler = 'standard',
#                                                   feature_names=feature_names)

