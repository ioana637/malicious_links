import os
from itertools import chain

import pandas as pd
from sklearn.linear_model import LogisticRegression

from utils import split_data, prediction, cal_metrics, appendMetricsTOCSV


def run_algorithm_lr_configuration(metrics, label, X, y,
                                   penalty='l2', dual=False, tol=1e-4,
                                   C=1.0, fit_intercept=True, intercept_scaling=1,
                                   class_weight='balanced', random_state=None, solver='lbfgs',
                                   max_iter=100, multi_class='auto', l1_ratio=None,
                                   stratify=False, train_size=0.8,
                                   normalize_data=False, scaler='min-max'
                                   ):
    X_train, X_test, y_train, y_test = split_data(X, y, normalize_data=normalize_data, stratify=stratify,
                                                  train_size=train_size, scaler=scaler);

    # Creating the classifier object
    classifier = LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=C,
                                    fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                                    class_weight=class_weight, random_state=random_state,
                                    solver=solver, max_iter=max_iter, multi_class=multi_class,
                                    l1_ratio=l1_ratio)

    # Performing training
    classifier.fit(X_train, y_train)

    # Make predictions
    y_pred, y_pred_probabilities = prediction(X_test, classifier)

    # Compute metrics
    precision, recall, f1, roc_auc = cal_metrics(y_test, y_pred, y_pred_probabilities, label)
    metrics['label'].append(label)

    metrics['penalty'].append(penalty)
    metrics['dual'].append(dual)
    metrics['tol'].append(tol)
    metrics['C'].append(C)
    metrics['fit_intercept'].append(fit_intercept)
    metrics['intercept_scaling'].append(intercept_scaling)
    metrics['class_weight'].append(class_weight)
    metrics['random_state'].append(random_state)
    metrics['solver'].append(solver)
    metrics['max_iter'].append(max_iter)
    metrics['multi_class'].append(multi_class)
    metrics['l1_ratio'].append(l1_ratio)

    metrics['precision'].append(precision)
    metrics['recall'].append(recall)
    metrics['f1_score'].append(f1)
    metrics['roc_auc'].append(roc_auc)


def init_metrics_for_LR():
    return {'label': [],
            'penalty': [], 'dual': [], 'tol': [], 'C': [], 'l1_ratio': [],
            'fit_intercept': [], 'intercept_scaling': [], 'class_weight':[],
            'random_state': [], 'solver': [], 'max_iter': [], 'multi_class': [],
            'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': []
            }


def run_algorithm_lr(df, filename='', stratify=False, train_size=0.8, normalize_data=False, scaler='min-max'):
    y = df['label'].copy()
    X = df.drop('label', axis=1).copy()
    metrics = init_metrics_for_LR()

    path_to_script = os.path.dirname(os.path.abspath(__file__))
    my_filename = os.path.join(path_to_script, 'lr', filename)

    # default algorithm
    run_algorithm_lr_configuration(metrics, 'LR - default params', X, y, stratify=stratify, train_size=train_size,
                                   normalize_data=normalize_data, scaler=scaler)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_LR, header=True)
    # CALIBRATING penalty
    run_algorithm_lr_configuration(metrics, 'LR: penalty = none', X, y, penalty='none', stratify=stratify,
                                   train_size=train_size, normalize_data=normalize_data, scaler=scaler)
    for l1_ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        run_algorithm_lr_configuration(metrics, 'LR: penalty = elasticnet, solver=saga, l1_ratio = ' + str(l1_ratio), X,
                                       y, solver='saga', l1_ratio=l1_ratio, penalty='elasticnet', stratify=stratify,
                                       train_size=train_size, normalize_data=normalize_data, scaler=scaler)
    run_algorithm_lr_configuration(metrics, 'LR: penalty = l1, solver = saga', X, y, solver='saga', penalty='l1',
                                   stratify=stratify, train_size=train_size, normalize_data=normalize_data,
                                   scaler=scaler)

    # CALIBRATING dual
    run_algorithm_lr_configuration(metrics, 'LR: dual = True, penalty = l2, solver = liblinear', X, y, penalty='l2',
                                   solver='liblinear', dual=True, stratify=stratify, train_size=train_size,
                                   normalize_data=normalize_data, scaler=scaler)

    # CALIBRATING tol
    for tol in [1e-2, 1e-3, 1e-5, 1e-6, 1e-7]:
        run_algorithm_lr_configuration(metrics, 'LR: tol = ' + str(tol), X, y, tol=tol, stratify=stratify,
                                       train_size=train_size, normalize_data=normalize_data, scaler=scaler)

    # CALIBRATE C
    for C in [0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2]:
        run_algorithm_lr_configuration(metrics, 'LR: C = ' + str(C), X, y, C=C, stratify=stratify,
                                       train_size=train_size, normalize_data=normalize_data, scaler=scaler)

    # CALIBRATE fit_intercept
    run_algorithm_lr_configuration(metrics, 'LR: fit_intercept=False', X, y, fit_intercept=False, stratify=stratify,
                                   train_size=train_size, normalize_data=normalize_data, scaler=scaler)

    # CALIBRATE intercept_scaling
    for intercept_scaling in range(2, 200, 10):
        run_algorithm_lr_configuration(metrics,
                                       'LR: solver = liblinear, fit_intercept=True, penalty = l2, intercept_scaling = ' + str(
                                           intercept_scaling), X, y, fit_intercept=True, solver='liblinear',
                                       intercept_scaling=intercept_scaling, penalty='l2', stratify=stratify,
                                       train_size=train_size, normalize_data=normalize_data, scaler=scaler)
        run_algorithm_lr_configuration(metrics,
                                       'LR: solver = liblinear, fit_intercept=True, penalty = l1, intercept_scaling = ' + str(
                                           intercept_scaling), X, y, fit_intercept=True, solver='liblinear',
                                       intercept_scaling=intercept_scaling, penalty='l1', stratify=stratify,
                                       train_size=train_size, normalize_data=normalize_data, scaler=scaler)

    # CALIBRATE class_weight
    run_algorithm_lr_configuration(metrics, 'LR: class_weight = balanced', X, y, class_weight='balanced',
                                   stratify=stratify, train_size=train_size, normalize_data=normalize_data,
                                   scaler=scaler)

    # CALIBRATE solver
    run_algorithm_lr_configuration(metrics, 'LR: solver=newton-cg, penalty = l2', X, y, penalty='l2',
                                   solver='newton-cg', stratify=stratify, train_size=train_size,
                                   normalize_data=normalize_data, scaler=scaler)
    run_algorithm_lr_configuration(metrics, 'LR: solver=newton-cg, penalty = none', X, y, penalty='none',
                                   solver='newton-cg', stratify=stratify, train_size=train_size,
                                   normalize_data=normalize_data, scaler=scaler)
    run_algorithm_lr_configuration(metrics, 'LR: solver=lbfgs, penalty = l2', X, y, penalty='l2', solver='lbfgs',
                                   stratify=stratify, train_size=train_size, normalize_data=normalize_data,
                                   scaler=scaler)
    run_algorithm_lr_configuration(metrics, 'LR: solver=lbfgs, penalty = none', X, y, penalty='none', solver='lbfgs',
                                   stratify=stratify, train_size=train_size, normalize_data=normalize_data,
                                   scaler=scaler)
    run_algorithm_lr_configuration(metrics, 'LR: solver=liblinear, penalty = l1', X, y, solver='liblinear',
                                   penalty='l1', stratify=stratify, train_size=train_size,
                                   normalize_data=normalize_data, scaler=scaler)
    run_algorithm_lr_configuration(metrics, 'LR: solver=liblinear, penalty = l2', X, y, solver='liblinear',
                                   penalty='l2', stratify=stratify, train_size=train_size,
                                   normalize_data=normalize_data, scaler=scaler)
    run_algorithm_lr_configuration(metrics, 'LR: solver=sag, penalty = l2', X, y, solver='sag', penalty='l2',
                                   stratify=stratify, train_size=train_size, normalize_data=normalize_data,
                                   scaler=scaler)
    run_algorithm_lr_configuration(metrics, 'LR: solver=sag, penalty = none', X, y, solver='sag', penalty='none',
                                   stratify=stratify, train_size=train_size, normalize_data=normalize_data,
                                   scaler=scaler)
    for l1_ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        run_algorithm_lr_configuration(metrics, 'LR: solver=saga, penalty = elasticnet, l1_ratio = ' + str(l1_ratio), X,
                                       y, l1_ratio=l1_ratio, solver='saga', penalty='elasticnet', stratify=stratify,
                                       train_size=train_size, normalize_data=normalize_data, scaler=scaler)
    run_algorithm_lr_configuration(metrics, 'LR: solver=saga, penalty = l1', X, y, solver='saga', penalty='l1',
                                   stratify=stratify, train_size=train_size, normalize_data=normalize_data,
                                   scaler=scaler)
    run_algorithm_lr_configuration(metrics, 'LR: solver=saga, penalty = l2', X, y, solver='saga', penalty='l2',
                                   stratify=stratify, train_size=train_size, normalize_data=normalize_data,
                                   scaler=scaler)
    run_algorithm_lr_configuration(metrics, 'LR: solver=saga, penalty = none', X, y, solver='saga', penalty='none',
                                   stratify=stratify, train_size=train_size, normalize_data=normalize_data,
                                   scaler=scaler)

    run_algorithm_lr_configuration(metrics, 'LR: solver=saga, penalty = none', X, y, solver='saga', penalty='none',
                                   stratify=stratify, train_size=train_size, normalize_data=normalize_data,
                                   scaler=scaler)
    run_algorithm_lr_configuration(metrics, 'LR: solver=saga, penalty = l1', X, y, solver='saga', penalty='l1',
                                   stratify=stratify, train_size=train_size, normalize_data=normalize_data,
                                   scaler=scaler)
    run_algorithm_lr_configuration(metrics, 'LR: solver=saga, penalty = l2', X, y, solver='saga', penalty='l2',
                                   stratify=stratify, train_size=train_size, normalize_data=normalize_data,
                                   scaler=scaler)

    run_algorithm_lr_configuration(metrics, 'LR: solver=sag, penalty = none', X, y, solver='sag', penalty='none',
                                   stratify=stratify, train_size=train_size, normalize_data=normalize_data,
                                   scaler=scaler)
    run_algorithm_lr_configuration(metrics, 'LR: solver=sag, penalty = l2', X, y, solver='sag', penalty='l2',
                                   stratify=stratify, train_size=train_size, normalize_data=normalize_data,
                                   scaler=scaler)

    # CALIBRATE max_iter
    for max_iter in range(10, 100, 10):
        run_algorithm_lr_configuration(metrics, 'LR: max_iter' + str(max_iter), X, y, max_iter=max_iter,
                                       stratify=stratify, train_size=train_size, normalize_data=normalize_data,
                                       scaler=scaler)
    for max_iter in range(110, 200, 10):
        run_algorithm_lr_configuration(metrics, 'LR: max_iter' + str(max_iter), X, y, max_iter=max_iter,
                                       stratify=stratify, train_size=train_size, normalize_data=normalize_data,
                                       scaler=scaler)
    for max_iter in range(210, 510, 50):
        run_algorithm_lr_configuration(metrics, 'LR: max_iter' + str(max_iter), X, y, max_iter=max_iter,
                                       stratify=stratify, train_size=train_size, normalize_data=normalize_data,
                                       scaler=scaler)

    # CALIBRATE multi_class
    run_algorithm_lr_configuration(metrics, 'LR: multi_class = ovr', X, y, multi_class='ovr', stratify=stratify,
                                   train_size=train_size, normalize_data=normalize_data, scaler=scaler)
    run_algorithm_lr_configuration(metrics, 'LR: multi_class = multinomial', X, y, multi_class='multinomial',
                                   stratify=stratify, train_size=train_size, normalize_data=normalize_data,
                                   scaler=scaler)

    # TODO: our proposed LR
    # sag and saga solvers work best with data normalization
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_LR)

    for max_iter in range(100, 300, 5):
        run_algorithm_lr_configuration(metrics, 'LR: solver=newton-cg, penalty = none, max_iter = ' + str(max_iter),
                                       X, y, max_iter=max_iter, solver='newton-cg', penalty='none',
                                       stratify=stratify, train_size=train_size, normalize_data=normalize_data,
                                       scaler=scaler)
        run_algorithm_lr_configuration(metrics, 'LR: solver=newton-cg, penalty = l2, max_iter = ' + str(max_iter),
                                       X, y, solver='newton-cg', penalty='l2', stratify=stratify,
                                       train_size=train_size, normalize_data=normalize_data, scaler=scaler)

        # CALIBRATING tol
        for tol in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
            run_algorithm_lr_configuration(metrics, 'LR: tol= ' + str(tol) + ', max_iter = ' + str(max_iter), X, y,
                                           tol=0.001, max_iter=max_iter, stratify=stratify, train_size=train_size,
                                           normalize_data=normalize_data, scaler=scaler)

        # CALIBRATE C
        for C in [0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2]:
            run_algorithm_lr_configuration(metrics, 'LR: C = ' + str(C) + ', max_iter = ' + str(max_iter), X, y,
                                           C=1e-07, max_iter=max_iter, stratify=stratify, train_size=train_size,
                                           normalize_data=normalize_data, scaler=scaler)

        for intercept_scaling in range(1, 300, 2):
            run_algorithm_lr_configuration(metrics,
                                           'LR: solver=liblinear, fit_intercept = True, penalty = l1, max_iter = ' + str(
                                               max_iter) + ',  intercept scaling = ' + str(intercept_scaling), X, y,
                                           fit_intercept=True, max_iter=max_iter, intercept_scaling=intercept_scaling,
                                           solver='liblinear', penalty='l1',
                                           stratify=stratify, train_size=train_size,
                                           normalize_data=normalize_data, scaler=scaler)
            run_algorithm_lr_configuration(metrics,
                                           'LR: solver=liblinear, fit_intercept = True, penalty = l2, max_iter = ' + str(
                                               max_iter) + ',  intercept scaling = ' + str(intercept_scaling), X, y,
                                           fit_intercept=True, solver='liblinear', penalty='l2', max_iter = max_iter,
                                           intercept_scaling=intercept_scaling, stratify=stratify,
                                           train_size=train_size, normalize_data=normalize_data, scaler=scaler)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_LR)

    for solver in ['newton-cg', 'liblinear']:
        for penalty in ['l1', 'l2']:
            concatenated_max_iter = chain(range(75, 85), range(105, 115), range(125, 145),
                                          range(155, 175), range(255, 265), range(355, 365))
            # 10 + 10 + 20 + 20 +10 +10 = 80
            for max_iter in concatenated_max_iter:
                concatenated_intercept_scaling = chain(range(7, 28), range(37, 78), range(97, 198))
                # 20 + 40 + 100 = 160
                for intercept_scaling in concatenated_intercept_scaling:
                    try:
                        run_algorithm_lr_configuration(metrics,
                                                   'LR: solver='+solver+', fit_intercept = True, penalty = '+penalty+', max_iter = ' + str(
                                                       max_iter) + ',  intercept scaling = ' + str(intercept_scaling), X, y,
                                                   fit_intercept=True, solver=solver, penalty=penalty, intercept_scaling=intercept_scaling,
                                                   max_iter = max_iter, stratify=stratify,
                                                   train_size=train_size, normalize_data=normalize_data, scaler=scaler)
                    except Exception as err:
                        print(err)
                metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_LR)



    # export metrics to CSV FILE
    # df_metrics = pd.DataFrame(metrics)
    # df_metrics.to_csv(my_filename, encoding='utf-8', index=True)
