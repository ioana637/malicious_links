import os
from itertools import chain
from multiprocessing import Manager, Pool

import pandas as pd
from sklearn.linear_model import LogisticRegression

from utils.data_post import compute_average_metric
from utils.data_pre import split_data_in_testing_training, load_normalized_dataset
from utils.utils import prediction, cal_metrics, appendMetricsTOCSV, convert_metrics_to_csv, listener_write_to_file


def create_LR_classifier(row):
    classifier = LogisticRegression(penalty=row['penalty'], dual=bool(row['dual']), tol=float(row['tol']),
                                    C=float(row['C']), fit_intercept=bool(row['dual']),
                                    intercept_scaling=int(row['intercept_scaling']),
                                    class_weight=row['class_weight'], random_state=None,
                                    solver=row['solver'],
                                    max_iter=int(row['max_iter']), multi_class=row['multi_class'],
                                    l1_ratio=None)
    return classifier


def run_algorithm_lr_configuration_parallel(X, y, q_metrics,
                                            penalty='l2', dual=False, tol=1e-4,
                                            C=1.0, fit_intercept=True, intercept_scaling=1,
                                            class_weight='balanced', random_state=None, solver='lbfgs',
                                            max_iter=100, multi_class='auto', l1_ratio=None,
                                            stratify=False, train_size=0.8
                                            ):
    X_train, X_test, y_train, y_test = split_data_in_testing_training(X, y, stratify, train_size)
    try:
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
        label = create_label_LR(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                                intercept_scaling=intercept_scaling, class_weight=class_weight,
                                random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class,
                                l1_ratio=l1_ratio)
        precision, recall, f1, roc_auc = cal_metrics(y_test, y_pred, y_pred_probabilities, label, classifier)
        string_results_for_queue = convert_metrics_to_csv(',', label, penalty, dual, tol, C, l1_ratio, fit_intercept,
                                                          intercept_scaling, class_weight, random_state, solver,
                                                          max_iter, multi_class, precision, recall, f1, roc_auc)
        q_metrics.put(string_results_for_queue)
    except Exception as er:
        # pass
        print(er)
        # traceback.print_exc()
        # print(traceback.format_exc())
    except RuntimeWarning as warn:
        # pass
        print(warn)


def run_algorithm_lr_configuration(metrics, label, X, y,
                                   penalty='l2', dual=False, tol=1e-4,
                                   C=1.0, fit_intercept=True, intercept_scaling=1,
                                   class_weight='balanced', random_state=None, solver='lbfgs',
                                   max_iter=100, multi_class='auto', l1_ratio=None,
                                   stratify=False, train_size=0.8
                                   ):
    X_train, X_test, y_train, y_test = split_data_in_testing_training(X, y, stratify, train_size)

    try:
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
        precision, recall, f1, roc_auc = cal_metrics(y_test, y_pred, y_pred_probabilities, label, classifier)
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
    except Exception as err:
        # pass
        print(err)
    except RuntimeWarning as warn:
        print(warn)
        # pass


def init_metrics_for_LR():
    return {'label': [],
            'penalty': [], 'dual': [], 'tol': [], 'C': [], 'l1_ratio': [],
            'fit_intercept': [], 'intercept_scaling': [], 'class_weight': [],
            'random_state': [], 'solver': [], 'max_iter': [], 'multi_class': [],
            'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': []
            }


def run_algorithm_lr(filename='', path='', stratify=False, train_size=0.8,
                     normalize_data=False, scaler='min-max'):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_LR()

    # full_path_filename = '/content/drive/MyDrive/code/' + filename
    # path_to_script = os.path.dirname(os.path.abspath(__file__))
    my_filename = os.path.join(path, 'results/lr', filename)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_LR, header=True)

    # default algorithm
    # run_algorithm_lr_configuration(metrics, 'LR - default params', X, y, stratify=stratify, train_size=train_size,
    #                                )
    # metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_LR)
    # # CALIBRATING penalty
    # run_algorithm_lr_configuration(metrics, 'LR: penalty = none', X, y, penalty='none', stratify=stratify,
    #                                train_size=train_size)
    # for l1_ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    #     run_algorithm_lr_configuration(metrics, 'LR: penalty = elasticnet, solver=saga, l1_ratio = ' + str(l1_ratio), X,
    #                                    y, solver='saga', l1_ratio=l1_ratio, penalty='elasticnet', stratify=stratify,
    #                                    train_size=train_size)
    # run_algorithm_lr_configuration(metrics, 'LR: penalty = l1, solver = saga', X, y, solver='saga', penalty='l1',
    #                                stratify=stratify, train_size=train_size)
    #
    # # CALIBRATING dual
    # run_algorithm_lr_configuration(metrics, 'LR: dual = True, penalty = l2, solver = liblinear', X, y, penalty='l2',
    #                                solver='liblinear', dual=True, stratify=stratify, train_size=train_size,
    #                                )
    #
    # # CALIBRATING tol
    # for tol in [1e-2, 1e-3, 1e-5, 1e-6, 1e-7]:
    #     run_algorithm_lr_configuration(metrics, 'LR: tol = ' + str(tol), X, y, tol=tol, stratify=stratify,
    #                                    train_size=train_size)
    #
    # # CALIBRATE C
    # for C in [0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2]:
    #     run_algorithm_lr_configuration(metrics, 'LR: C = ' + str(C), X, y, C=C, stratify=stratify,
    #                                    train_size=train_size)
    #
    # # CALIBRATE fit_intercept
    # run_algorithm_lr_configuration(metrics, 'LR: fit_intercept=False', X, y, fit_intercept=False, stratify=stratify,
    #                                train_size=train_size)
    #
    # # CALIBRATE intercept_scaling
    # for intercept_scaling in range(2, 200, 10):
    #     run_algorithm_lr_configuration(metrics,
    #                                    'LR: solver = liblinear, fit_intercept=True, penalty = l2, intercept_scaling = ' + str(
    #                                        intercept_scaling), X, y, fit_intercept=True, solver='liblinear',
    #                                    intercept_scaling=intercept_scaling, penalty='l2', stratify=stratify,
    #                                    train_size=train_size)
    #     run_algorithm_lr_configuration(metrics,
    #                                    'LR: solver = liblinear, fit_intercept=True, penalty = l1, intercept_scaling = ' + str(
    #                                        intercept_scaling), X, y, fit_intercept=True, solver='liblinear',
    #                                    intercept_scaling=intercept_scaling, penalty='l1', stratify=stratify,
    #                                    train_size=train_size)
    # # CALIBRATE class_weight
    # run_algorithm_lr_configuration(metrics, 'LR: class_weight = balanced', X, y, class_weight='balanced',
    #                                stratify=stratify, train_size=train_size)
    #
    # # CALIBRATE solver
    # run_algorithm_lr_configuration(metrics, 'LR: solver=newton-cg, penalty = l2', X, y, penalty='l2',
    #                                solver='newton-cg', stratify=stratify, train_size=train_size,
    #                                )
    # run_algorithm_lr_configuration(metrics, 'LR: solver=newton-cg, penalty = none', X, y, penalty='none',
    #                                solver='newton-cg', stratify=stratify, train_size=train_size,
    #                                )
    # run_algorithm_lr_configuration(metrics, 'LR: solver=lbfgs, penalty = l2', X, y, penalty='l2', solver='lbfgs',
    #                                stratify=stratify, train_size=train_size)
    # run_algorithm_lr_configuration(metrics, 'LR: solver=lbfgs, penalty = none', X, y, penalty='none', solver='lbfgs',
    #                                stratify=stratify, train_size=train_size)
    # run_algorithm_lr_configuration(metrics, 'LR: solver=liblinear, penalty = l1', X, y, solver='liblinear',
    #                                penalty='l1', stratify=stratify, train_size=train_size)
    # run_algorithm_lr_configuration(metrics, 'LR: solver=liblinear, penalty = l2', X, y, solver='liblinear',
    #                                penalty='l2', stratify=stratify, train_size=train_size)
    # run_algorithm_lr_configuration(metrics, 'LR: solver=sag, penalty = l2', X, y, solver='sag', penalty='l2',
    #                                stratify=stratify, train_size=train_size)
    # run_algorithm_lr_configuration(metrics, 'LR: solver=sag, penalty = none', X, y, solver='sag', penalty='none',
    #                                stratify=stratify, train_size=train_size)
    # for l1_ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    #     run_algorithm_lr_configuration(metrics, 'LR: solver=saga, penalty = elasticnet, l1_ratio = ' + str(l1_ratio), X,
    #                                    y, l1_ratio=l1_ratio, solver='saga', penalty='elasticnet', stratify=stratify,
    #                                    train_size=train_size)
    # run_algorithm_lr_configuration(metrics, 'LR: solver=saga, penalty = l1', X, y, solver='saga', penalty='l1',
    #                                stratify=stratify, train_size=train_size)
    # run_algorithm_lr_configuration(metrics, 'LR: solver=saga, penalty = l2', X, y, solver='saga', penalty='l2',
    #                                stratify=stratify, train_size=train_size)
    # run_algorithm_lr_configuration(metrics, 'LR: solver=saga, penalty = none', X, y, solver='saga', penalty='none',
    #                                stratify=stratify, train_size=train_size)
    #
    # run_algorithm_lr_configuration(metrics, 'LR: solver=saga, penalty = none', X, y, solver='saga', penalty='none',
    #                                stratify=stratify, train_size=train_size)
    # run_algorithm_lr_configuration(metrics, 'LR: solver=saga, penalty = l1', X, y, solver='saga', penalty='l1',
    #                                stratify=stratify, train_size=train_size)
    # run_algorithm_lr_configuration(metrics, 'LR: solver=saga, penalty = l2', X, y, solver='saga', penalty='l2',
    #                                stratify=stratify, train_size=train_size)
    #
    # run_algorithm_lr_configuration(metrics, 'LR: solver=sag, penalty = none', X, y, solver='sag', penalty='none',
    #                                stratify=stratify, train_size=train_size)
    # run_algorithm_lr_configuration(metrics, 'LR: solver=sag, penalty = l2', X, y, solver='sag', penalty='l2',
    #                                stratify=stratify, train_size=train_size)
    #
    # # CALIBRATE max_iter
    # for max_iter in range(10, 100, 10):
    #     run_algorithm_lr_configuration(metrics, 'LR: max_iter' + str(max_iter), X, y, max_iter=max_iter,
    #                                    stratify=stratify, train_size=train_size)
    # for max_iter in range(110, 200, 10):
    #     run_algorithm_lr_configuration(metrics, 'LR: max_iter' + str(max_iter), X, y, max_iter=max_iter,
    #                                    stratify=stratify, train_size=train_size)
    # for max_iter in range(210, 510, 50):
    #     run_algorithm_lr_configuration(metrics, 'LR: max_iter' + str(max_iter), X, y, max_iter=max_iter,
    #                                    stratify=stratify, train_size=train_size)
    #
    # # CALIBRATE multi_class
    # run_algorithm_lr_configuration(metrics, 'LR: multi_class = ovr', X, y, multi_class='ovr', stratify=stratify,
    #                                train_size=train_size)
    # run_algorithm_lr_configuration(metrics, 'LR: multi_class = multinomial', X, y, multi_class='multinomial',
    #                                stratify=stratify, train_size=train_size)
    #
    # # sag and saga solvers work best with data normalization
    # metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_LR)
    #
    # for max_iter in range(100, 300, 5):
    #     run_algorithm_lr_configuration(metrics, 'LR: solver=newton-cg, penalty = none, max_iter = ' + str(max_iter),
    #                                    X, y, max_iter=max_iter, solver='newton-cg', penalty='none',
    #                                    stratify=stratify, train_size=train_size)
    #     run_algorithm_lr_configuration(metrics, 'LR: solver=newton-cg, penalty = l2, max_iter = ' + str(max_iter),
    #                                    X, y, solver='newton-cg', penalty='l2', stratify=stratify,
    #                                    train_size=train_size)
    #
    #     # CALIBRATING tol
    #     for tol in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    #         run_algorithm_lr_configuration(metrics, 'LR: tol= ' + str(tol) + ', max_iter = ' + str(max_iter), X, y,
    #                                        tol=0.001, max_iter=max_iter, stratify=stratify, train_size=train_size,
    #                                        )
    #
    #     # CALIBRATE C
    #     for C in [0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2]:
    #         run_algorithm_lr_configuration(metrics, 'LR: C = ' + str(C) + ', max_iter = ' + str(max_iter), X, y,
    #                                        C=1e-07, max_iter=max_iter, stratify=stratify, train_size=train_size,
    #                                        )
    #
    #     for intercept_scaling in range(1, 300, 2):
    #         run_algorithm_lr_configuration(metrics,
    #                                        'LR: solver=liblinear, fit_intercept = True, penalty = l1, max_iter = ' + str(
    #                                            max_iter) + ',  intercept scaling = ' + str(intercept_scaling), X, y,
    #                                        fit_intercept=True, max_iter=max_iter, intercept_scaling=intercept_scaling,
    #                                        solver='liblinear', penalty='l1',
    #                                        stratify=stratify, train_size=train_size)
    #         run_algorithm_lr_configuration(metrics,
    #                                        'LR: solver=liblinear, fit_intercept = True, penalty = l2, max_iter = ' + str(
    #                                            max_iter) + ',  intercept scaling = ' + str(intercept_scaling), X, y,
    #                                        fit_intercept=True, solver='liblinear', penalty='l2', max_iter = max_iter,
    #                                        intercept_scaling=intercept_scaling, stratify=stratify,
    #                                        train_size=train_size)
    # metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_LR)

    for solver in ['newton-cg', 'liblinear']:
        for penalty in ['l1', 'l2']:
            for max_iter in range(100, 300, 1):
                concatenated_intercept_scaling = chain(range(10, 70, 1), range(100, 115, 1), range(165, 209, 1),
                                                       range(223, 300, 1))
                for intercept_scaling in concatenated_intercept_scaling:
                    try:
                        run_algorithm_lr_configuration(metrics,
                                                       'LR: solver=' + solver + ', fit_intercept = True, penalty = ' + penalty + ', max_iter = ' + str(
                                                           max_iter) + ',  intercept scaling = ' + str(
                                                           intercept_scaling), X, y,
                                                       fit_intercept=True, solver=solver, penalty=penalty,
                                                       intercept_scaling=intercept_scaling,
                                                       max_iter=max_iter, class_weight='balanced', multi_class='auto',
                                                       stratify=stratify, train_size=train_size, )
                    except Exception as err:
                        print(err)
                metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_LR)

    # export metrics to CSV FILE
    # df_metrics = pd.DataFrame(metrics)
    # df_metrics.to_csv(my_filename, encoding='utf-8', index=True)


def run_algorithm_lr_parallel(filename='', path='', stratify=False, train_size=0.8,
                              normalize_data=False, scaler='min-max', no_threads=8):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_LR()
    my_filename = os.path.join(path, 'results/lr', filename)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_LR, header=True)

    solver_list = ['newton-cg', 'liblinear']
    penalty_list = ['l1', 'l2']
    max_iter_list = list(range(100, 300, 1))
    intercept_scaling_list = list(chain(range(10, 70, 1), range(100, 115, 1), range(165, 209, 1), range(223, 300, 1)))
    # 156800 configs

    with Manager() as manager:
        q_metrics = manager.Queue()
        jobs = []

        with Pool(no_threads) as pool:
            watcher = pool.apply_async(listener_write_to_file, (q_metrics, my_filename))
            for solver in solver_list:
                for penalty in penalty_list:
                    for max_iter in max_iter_list:
                        for intercept_scaling in intercept_scaling_list:
                            job = pool.apply_async(run_algorithm_lr_configuration_parallel,
                                                   (X, y, q_metrics,
                                                    penalty, False, 1e-4, 1.0, True, intercept_scaling, 'balanced',
                                                    None, solver, max_iter, 'auto', None,
                                                    stratify, train_size))
                            # print(job)
                            jobs.append(job)

            # print(jobs)
            # collect results from the workers through the pool result queue
            for job in jobs:
                job.get()
            # now we are done, kill the listener
            q_metrics.put('kill')
            pool.close()
            pool.join()


def run_best_configs_lr(df_configs, filename='', path='', stratify=True, train_size=0.8,
                        normalize_data=True, scaler='min-max', n_rep=100):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_LR()
    my_filename = os.path.join(path, 'new_results\\lr', filename)

    for i in range(1, n_rep):
        for index, row in df_configs.iterrows():
            label = create_label_LR_for_row(row)
            run_algorithm_lr_configuration(metrics, label, X, y,
                                           penalty=row['penalty'], dual=bool(row['dual']), tol=float(row['tol']),
                                           C=float(row['C']), fit_intercept=bool(row['dual']),
                                           intercept_scaling=int(row['intercept_scaling']),
                                           class_weight=row['class_weight'], random_state=None,
                                           solver=row['solver'],
                                           max_iter=int(row['max_iter']), multi_class=row['multi_class'],
                                           l1_ratio=None,
                                           stratify=stratify, train_size=train_size
                                           )

    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.groupby(['label'], as_index=False).agg({'precision': 'mean', 'recall': 'mean',
                                                                    'f1_score': 'mean', 'roc_auc': 'mean',
                                                                    'penalty': 'first',
                                                                    'dual': 'first',
                                                                    'tol': 'first',
                                                                    'fit_intercept': 'first',
                                                                    'intercept_scaling': 'first',
                                                                    'class_weight': 'first',
                                                                    'random_state': 'first',
                                                                    'solver': 'first',
                                                                    'max_iter': 'first',
                                                                    'multi_class': 'first',
                                                                    'l1_ratio': 'first',
                                                                    'C': 'first'})
    metrics_df = compute_average_metric(metrics_df)
    metrics_df.sort_values(by=['average_metric'], ascending=False, inplace=True)
    metrics = appendMetricsTOCSV(my_filename, metrics_df, init_metrics_for_LR, header=True)


def create_label_LR_for_row(row):
    return create_label_LR(row['penalty'], row['dual'], row['tol'], row['C'], row['fit_intercept'],
                           row['intercept_scaling'], row['class_weight'], row['random_state'], row['solver'],
                           row['max_iter'], row['multi_class'], row['l1_ratio'])


def create_label_LR(penalty, dual, tol, C, fit_intercept, intercept_scaling, class_weight, random_state, solver,
                    max_iter, multi_class, l1_ratio):
    label = "Logistic Regression, penalty=" + str(
        penalty) + ", dual=" + str(dual) + ", tol=" + str(tol) + ", C=" + str(C) + ", fit_intercept=" + str(
        fit_intercept) + ", intercept_scaling=" + str(
        intercept_scaling) + ", class_weight=" + class_weight + ", random_state" + str(
        random_state) + ", solver=" + solver + ', max_iter=' + str(
        max_iter) + ', multi_class=' + multi_class + ", l1_ratio=" + \
            str(l1_ratio)
    return label
