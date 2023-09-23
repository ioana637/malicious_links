import os
import pandas as pd

from algorithms.dt import init_metrics_for_DT, run_algorithm_dt_configuration_feature_importance
from algorithms.rfc import init_metrics_for_rfc, run_algorithm_rfc_configuration_feature_importance

run_algorithm_rfc_configuration_feature_importance(criterion='entropy', n_estimators=110,
                                                   min_samples_leaf=3, min_samples_split=4, max_depth=53,
                                                   max_leaf_nodes=1564, max_features='sqrt',
                                                   min_weight_fraction_leaf=0.0,
                                                   min_impurity_decrease=0.0, bootstrap=True, oob_score=False,
                                                   class_weight='balanced', ccp_alpha=0.0, max_samples=None,
                                                   stratify=True, train_size=0.8,
                                                   normalize_data=True, scaler='min-max')

run_algorithm_dt_configuration_feature_importance(criterion='entropy',
                                                  splitter='best', max_depth=8, min_samples_leaf=1,
                                                  min_samples_split=7, min_weight_fraction_leaf=0.0,
                                                  max_features=None, max_leaf_nodes=20, min_impurity_decrease=0.0,
                                                  class_weight='balanced', ccp_alpha=0.0,
                                                  train_size=0.8, stratify=True,
                                                  normalize_data=True, scaler='standard')
