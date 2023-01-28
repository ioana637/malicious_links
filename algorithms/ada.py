import os
from itertools import chain
import warnings

from sklearn.ensemble import AdaBoostClassifier

warnings.filterwarnings("error")

import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, RBF, Matern, RationalQuadratic, DotProduct

from utils import split_data, prediction, cal_metrics, appendMetricsTOCSV


def run_algorithm_ada_boost_configuration(metrics, label, X, y,
                                         base_estimator=None,
                                         n_estimators=50,
                                         learning_rate=1.0,
                                         algorithm='SAMME.R',
                                         stratify=False, train_size=0.8,
                                         normalize_data=False, scaler='min-max'
                                         ):
    X_train, X_test, y_train, y_test = split_data(X, y, normalize_data=normalize_data, stratify=stratify,
                                                  train_size=train_size, scaler=scaler)
    try:
        # Creating the classifier object
        classifier = AdaBoostClassifier(base_estimator = base_estimator, n_estimators = n_estimators,
                                        learning_rate = learning_rate, algorithm = algorithm)
        # Performing training
        classifier.fit(X_train, y_train)

        # Make predictions
        y_pred, y_pred_probabilities = prediction(X_test, classifier)

        # Compute metrics
        precision, recall, f1, roc_auc = cal_metrics(y_test, y_pred, y_pred_probabilities, label)
        metrics['label'].append(label)

        metrics['base_estimator'].append(base_estimator)
        metrics['n_estimators'].append(n_estimators)
        metrics['learning_rate'].append(learning_rate)
        metrics['algorithm'].append(algorithm)

        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1_score'].append(f1)
        metrics['roc_auc'].append(roc_auc)
    except Exception as er:
        # print(er)
        pass
    except RuntimeWarning as warn:
        # print(warn)
        pass

def init_metrics_for_AdaBoost():
    return {'label': [],
            'base_estimator': [], 'n_estimators': [],
            'learning_rate': [], 'algorithm': [],
            'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': []
            }


def run_algorithm_ada_boost(df, filename='', stratify=False, train_size=0.8, normalize_data=False, scaler='min-max'):
    y = df['label'].copy()
    X = df.drop('label', axis=1).copy()
    metrics = init_metrics_for_AdaBoost()

    path_to_script = os.path.dirname(os.path.abspath(__file__))
    my_filename = os.path.join(path_to_script, 'ada', filename)

    # default algorithm
    run_algorithm_ada_boost_configuration(metrics, 'Ada Boost - default params', X, y, stratify = stratify,
                                          train_size = train_size, normalize_data=normalize_data,
                                          scaler=scaler)
    run_algorithm_ada_boost_configuration(metrics, 'Ada Boost - algorithm = SAMME', X, y, algorithm = 'SAMME',
                                          stratify = stratify, train_size = train_size,
                                          normalize_data=normalize_data, scaler= scaler)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_AdaBoost, header=True)
    # CALIBRATING n_estimators
    for n_estimators in range(1, 100, 2):
        run_algorithm_ada_boost_configuration(metrics, 'Ada Boost - n_estimators = ' + str(n_estimators), X, y,
                                              n_estimators = n_estimators, stratify = stratify,
                                              train_size = train_size, normalize_data=normalize_data, scaler=scaler)
        run_algorithm_ada_boost_configuration(metrics, 'Ada Boost - algorithm = SAMME, n_estimators = ' +
                                              str(n_estimators), X, y,algorithm = 'SAMME',
                                              n_estimators = n_estimators, stratify = stratify,
                                              train_size = train_size, normalize_data=normalize_data, scaler=scaler)
    for n_estimators in range(100, 200, 5):
        run_algorithm_ada_boost_configuration(metrics, 'Ada Boost - n_estimators = ' + str(n_estimators), X, y,
                                              n_estimators = n_estimators, stratify = stratify,
                                              train_size = train_size, normalize_data=normalize_data, scaler=scaler)
        run_algorithm_ada_boost_configuration(metrics, 'Ada Boost - algorithm = SAMME, n_estimators = ' +
                                              str(n_estimators), X, y,algorithm = 'SAMME',
                                              n_estimators = n_estimators, stratify = stratify,
                                              train_size = train_size, normalize_data=normalize_data, scaler=scaler)
    for n_estimators in range(200, 300, 10):
        run_algorithm_ada_boost_configuration(metrics, 'Ada Boost - n_estimators = ' + str(n_estimators), X, y,
                                              n_estimators = n_estimators, stratify = stratify,
                                              train_size = train_size, normalize_data=normalize_data, scaler=scaler)
        run_algorithm_ada_boost_configuration(metrics, 'Ada Boost - algorithm = SAMME, n_estimators = ' +
                                              str(n_estimators), X, y,algorithm = 'SAMME',
                                              n_estimators = n_estimators, stratify = stratify,
                                              train_size = train_size, normalize_data=normalize_data, scaler=scaler)
    for n_estimators in range(300, 500, 25):
        run_algorithm_ada_boost_configuration(metrics, 'Ada Boost - n_estimators = ' + str(n_estimators), X, y,
                                              n_estimators = n_estimators, stratify = stratify,
                                              train_size = train_size, normalize_data=normalize_data, scaler=scaler)
        run_algorithm_ada_boost_configuration(metrics, 'Ada Boost -  algorithm = SAMME, n_estimators = ' +
                                              str(n_estimators), X, y,algorithm = 'SAMME',
                                              n_estimators = n_estimators, stratify = stratify,
                                              train_size = train_size, normalize_data=normalize_data, scaler=scaler)
    for n_estimators in range(500, 1000, 50):
        run_algorithm_ada_boost_configuration(metrics, 'Ada Boost - n_estimators = ' + str(n_estimators), X, y,
                                              n_estimators = n_estimators, stratify = stratify,
                                              train_size = train_size, normalize_data=normalize_data, scaler=scaler)
        run_algorithm_ada_boost_configuration(metrics, 'Ada Boost - algorithm = SAMME, n_estimators = ' +
                                              str(n_estimators), X, y,algorithm = 'SAMME',
                                              n_estimators = n_estimators, stratify = stratify,
                                              train_size = train_size, normalize_data=normalize_data, scaler=scaler)

        # CALIBRATING learning_rate
    for learning_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                          ]:
        run_algorithm_ada_boost_configuration(metrics, 'Ada Boost - learning_rate = ' + str(learning_rate),
                                              X, y, learning_rate = learning_rate, stratify = stratify,
                                              train_size = train_size, normalize_data=normalize_data, scaler=scaler)
        run_algorithm_ada_boost_configuration(metrics, 'Ada Boost - algorithm = SAMME, learning_rate = ' +
                                              str(learning_rate), X, y,algorithm = 'SAMME',
                                              learning_rate = learning_rate, stratify = stratify,
                                              train_size = train_size, normalize_data=normalize_data, scaler=scaler)

    for algorithm in ['SAMME', 'SAMME.R']:
        for learning_rate in np.random.uniform(low=0.0001, high=1.9999, size=(20,)):
            for n_estimators in chain(range(37, 58, 2), range(64, 115, 2), range(165, 180, 2),
                                      range(190, 225, 2), range(235, 245, 2), range(275, 305, 2),
                                      range(370, 380, 2), range(420, 430, 2), range(445, 455, 2), range(495, 505, 2),
                                      range(470, 480, 2), range(545, 555, 2), range(645, 655, 2), range(795, 805, 2),
                                      range(845, 855, 2), range(895, 905, 2), range(945, 955, 2)):
            #     10 +25 +8+18+15+12*5 = 70+40+25=135
                run_algorithm_ada_boost_configuration(metrics, 'Ada Boost - learning_rate = ' + str(learning_rate)+
                                                  ', algorithm='+algorithm+', n_estimators='+
                                                  str(n_estimators), X, y, learning_rate = learning_rate,
                                                  algorithm=algorithm, n_estimators=n_estimators,
                                                  stratify = stratify, train_size = train_size,
                                                  normalize_data=normalize_data, scaler=scaler)
            metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_AdaBoost)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_AdaBoost)
