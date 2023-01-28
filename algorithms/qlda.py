import os

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import warnings
warnings.filterwarnings("error")
from utils import split_data, prediction, cal_metrics, appendMetricsTOCSV


def run_algorithm_qda_configuration(metrics, label, X, y, tol=1e-4,
                                    stratify=False, train_size=0.8,
                                    normalize_data=False, scaler='min-max'
                                    ):
    X_train, X_test, y_train, y_test = split_data(X, y, normalize_data=normalize_data, stratify=stratify,
                                                  train_size=train_size, scaler=scaler);
    try:
        # Creating the classifier object
        classifier = QuadraticDiscriminantAnalysis(tol=tol)

        # Performing training
        classifier.fit(X_train, y_train)

        # Make predictions
        y_pred, y_pred_probabilities = prediction(X_test, classifier)

        # Compute metrics
        precision, recall, f1, roc_auc = cal_metrics(y_test, y_pred, y_pred_probabilities, label)
        metrics['label'].append(label)

        metrics['tol'].append(tol)

        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1_score'].append(f1)
        metrics['roc_auc'].append(roc_auc)
    except Exception as err:
        print(err)
    except RuntimeWarning as warn:
        print(warn)



def init_metrics_for_QDA():
    return {'label': [], 'tol': [],
            'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': []
            }


def run_algorithm_qda(df, filename='', stratify=False, train_size=0.8, normalize_data=False, scaler='min-max'):
    y = df['label'].copy()
    X = df.drop('label', axis=1).copy()
    metrics = init_metrics_for_QDA()

    path_to_script = os.path.dirname(os.path.abspath(__file__))
    my_filename = os.path.join(path_to_script, 'qlda', filename)

    # default algorithm
    run_algorithm_qda_configuration(metrics, 'QDA - default params', X, y, stratify=stratify, train_size=train_size,
                                    normalize_data=normalize_data, scaler=scaler)

    # CALIBRATING tol
    for tol in [1e-2, 1e-3, 1e-5, 1e-6, 1e-7]:
        run_algorithm_qda_configuration(metrics, 'QDA: tol = ' + str(tol), X, y, tol=tol,
                                        stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_QDA, header=True)
    for tol in np.random.uniform(low=1e-7, high=0.02, size=(1000,)):
        run_algorithm_qda_configuration(metrics, 'QDA: tol = ' + str(tol), X, y, tol=tol,
                                        stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_QDA)


def run_algorithm_lda_configuration(metrics, label, X, y, tol=1e-4,
                                    solver='svd', shrinkage=None, store_covariance=False,
                                    n_components=None, covariance_estimator=None,
                                    stratify=False, train_size=0.8,
                                    normalize_data=False, scaler='min-max'
                                    ):
    X_train, X_test, y_train, y_test = split_data(X, y, normalize_data=normalize_data, stratify=stratify,
                                                  train_size=train_size, scaler=scaler)
    try:
        # Creating the classifier object
        classifier = LinearDiscriminantAnalysis(tol=tol, solver=solver, shrinkage=shrinkage,
                                                n_components=n_components, store_covariance=store_covariance,
                                                covariance_estimator=covariance_estimator)

        # Performing training
        classifier.fit(X_train, y_train)

        # Make predictions
        y_pred, y_pred_probabilities = prediction(X_test, classifier)

        # Compute metrics
        precision, recall, f1, roc_auc = cal_metrics(y_test, y_pred, y_pred_probabilities, label)
        metrics['label'].append(label)

        metrics['tol'].append(tol)
        metrics['solver'].append(solver)
        metrics['shrinkage'].append(shrinkage)
        metrics['n_components'].append(n_components)
        metrics['store_covariance'].append(store_covariance)
        metrics['covariance_estimator'].append(covariance_estimator)

        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1_score'].append(f1)
        metrics['roc_auc'].append(roc_auc)
    except Exception as err:
        print(err)
    except RuntimeWarning as warn:
        print(warn)


def init_metrics_for_LDA():
    return {'label': [], 'tol': [], 'solver': [], 'shrinkage': [], 'n_components': [],
            'store_covariance': [], 'covariance_estimator': [],
            'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': []
            }


def run_algorithm_lda(df, filename='', stratify=False, train_size=0.8, normalize_data=False, scaler='min-max'):
    y = df['label'].copy()
    X = df.drop('label', axis=1).copy()
    metrics = init_metrics_for_LDA()

    path_to_script = os.path.dirname(os.path.abspath(__file__))
    my_filename = os.path.join(path_to_script, 'qlda', filename)

    # default algorithm
    run_algorithm_lda_configuration(metrics, 'LDA - default params', X, y, stratify=stratify, train_size=train_size,
                                    normalize_data=normalize_data, scaler=scaler)

    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_LDA, header=True)
    # CALIBRATING tol
    for tol in [1e-2, 1e-3, 1e-5, 1e-6, 1e-7]:
        run_algorithm_lda_configuration(metrics, 'LDA: tol = ' + str(tol), X, y, tol=tol,
                                        stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)

    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_LDA)

    # CALIBRATING solver
    run_algorithm_lda_configuration(metrics, 'LDA: solver = lsqr', X, y, solver='lsqr', stratify=stratify,
                                    train_size=train_size, normalize_data=normalize_data, scaler=scaler)

    # CALIBRATING shrinkage
    run_algorithm_lda_configuration(metrics, 'LDA: solver = lsqr, shrinkage = auto', X, y, solver='lsqr',
                                    shrinkage='auto', stratify=stratify, train_size=train_size,
                                    normalize_data=normalize_data, scaler=scaler)
    run_algorithm_lda_configuration(metrics, 'LDA: solver = eigen, shrinkage = auto', X, y, solver='eigen',
                                    shrinkage='auto', stratify=stratify, train_size=train_size,
                                    normalize_data=normalize_data, scaler=scaler)
    for shrinkage in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        run_algorithm_lda_configuration(metrics, 'LDA: solver = eigen, shrinkage = ' + str(shrinkage), X, y,
                                        solver='eigen', shrinkage=shrinkage, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_lda_configuration(metrics, 'LDA: solver = lsqr, shrinkage = ' + str(shrinkage), X, y,
                                        solver='lsqr', shrinkage=shrinkage, stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)

    # CALIBRATING n_components
    run_algorithm_lda_configuration(metrics, 'LDA: n_components = 0', X, y, n_components=0, stratify=stratify,
                                    train_size=train_size, normalize_data=normalize_data, scaler=scaler)
    run_algorithm_lda_configuration(metrics, 'LDA: n_components = 1', X, y, n_components=1, stratify=stratify,
                                    train_size=train_size, normalize_data=normalize_data, scaler=scaler)

    # CALIBRATING store_covariance
    run_algorithm_lda_configuration(metrics, 'LDA: store_covariance = True, solver = svd', X, y, solver='svd',
                                    store_covariance=True, stratify=stratify, train_size=train_size,
                                    normalize_data=normalize_data, scaler=scaler)

    # CALIBRATING covariance_estimator
    from sklearn.covariance import EmpiricalCovariance, EllipticEnvelope
    from sklearn.covariance import GraphicalLassoCV

    # EmpiricalCovariance -------calibrating params
    run_algorithm_lda_configuration(metrics, 'LDA: covariance_estimator = EmpiricalCovariance, solver = lsqr', X, y,
                                    solver='lsqr', covariance_estimator=EmpiricalCovariance(), stratify=stratify,
                                    train_size=train_size, normalize_data=normalize_data, scaler=scaler)
    run_algorithm_lda_configuration(metrics,
                                    'LDA: covariance_estimator = EmpiricalCovariance(assume_centered = True), solver = lsqr',
                                    X, y, solver='lsqr', covariance_estimator=EmpiricalCovariance(assume_centered=True),
                                    stratify=stratify, train_size=train_size,
                                    normalize_data=normalize_data, scaler=scaler)

    # EllipticEnvelope ----- calibrating params as well
    run_algorithm_lda_configuration(metrics,
                                    'LDA: covariance_estimator = EllipticEnvelope(random_state=0), solver = lsqr', X, y,
                                    solver='lsqr', covariance_estimator=EllipticEnvelope(random_state=0),
                                    stratify=stratify, train_size=train_size,
                                    normalize_data=normalize_data, scaler=scaler)
    run_algorithm_lda_configuration(metrics,
                                    'LDA: covariance_estimator = EllipticEnvelope(random_state=0, assume_centered = True), solver = lsqr',
                                    X, y, solver='lsqr',
                                    covariance_estimator=EllipticEnvelope(random_state=0, assume_centered=True),
                                    stratify=stratify, train_size=train_size,
                                    normalize_data=normalize_data, scaler=scaler)

    for support_fraction in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        run_algorithm_lda_configuration(metrics,
                                        'LDA: covariance_estimator = EllipticEnvelope(random_state=0, support_fraction = ' + str(
                                            support_fraction) + '), solver = lsqr', X, y, solver='lsqr',
                                        covariance_estimator=EllipticEnvelope(random_state=0,
                                                                              support_fraction=support_fraction),
                                        stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)

    for contamination in [0.1, 0.2, 0.3, 0.4, 0.5]:
        run_algorithm_lda_configuration(metrics,
                                        'LDA: covariance_estimator = EllipticEnvelope(random_state=0, contamination = ' + str(
                                            contamination) + '), solver = lsqr', X, y, solver='lsqr',
                                        covariance_estimator=EllipticEnvelope(random_state=0,
                                                                              contamination=contamination),
                                        stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)

    # GraphicalLassoCV ----------- with params
    run_algorithm_lda_configuration(metrics, 'LDA: covariance_estimator = GraphicalLassoCV, solver = lsqr', X, y,
                                    solver='lsqr',
                                    covariance_estimator=GraphicalLassoCV(), stratify=stratify, train_size=train_size,
                                    normalize_data=normalize_data, scaler=scaler)

    for alphas in range(1, 4, 1):
        run_algorithm_lda_configuration(metrics, 'LDA: covariance_estimator = GraphicalLassoCV(n_refinements = ' + str(
            alphas) + '), solver = lsqr',
                                        X, y, solver='lsqr',
                                        covariance_estimator=GraphicalLassoCV(n_refinements=alphas),
                                        stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)

    for alphas in range(5, 15, 1):
        run_algorithm_lda_configuration(metrics, 'LDA: covariance_estimator = GraphicalLassoCV(alphas = ' + str(
            alphas) + '), solver = lsqr',
                                        X, y, solver='lsqr', covariance_estimator=GraphicalLassoCV(alphas=alphas),
                                        stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_lda_configuration(metrics,
                                        'LDA: covariance_estimator = GraphicalLassoCV(mode = lars, alphas = ' + str(
                                            alphas) + '), solver = lsqr', X, y, solver='lsqr',
                                        covariance_estimator=GraphicalLassoCV(mode='lars', alphas=alphas),
                                        stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_lda_configuration(metrics,
                                        'LDA: covariance_estimator = GraphicalLassoCV(assume_centered = True, alphas = ' + str(
                                            alphas) + '), solver = lsqr', X, y, solver='lsqr',
                                        covariance_estimator=GraphicalLassoCV(assume_centered=True, alphas=alphas),
                                        stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_lda_configuration(metrics,
                                        'LDA: covariance_estimator = GraphicalLassoCV(mode = lars, assume_centered = True, alphas = ' + str(
                                            alphas) + '), solver = lsqr', X, y, solver='lsqr',
                                        covariance_estimator=GraphicalLassoCV(mode='lars', assume_centered=True,
                                                                              alphas=alphas), stratify=stratify,
                                        train_size=train_size, normalize_data=normalize_data, scaler=scaler)

        run_algorithm_lda_configuration(metrics, 'LDA: covariance_estimator = GraphicalLassoCV(n_refinements = ' + str(
            alphas) + '), solver = lsqr',
                                        X, y, solver='lsqr',
                                        covariance_estimator=GraphicalLassoCV(n_refinements=alphas),
                                        stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_lda_configuration(metrics,
                                        'LDA: covariance_estimator = GraphicalLassoCV(mode = lars, n_refinements = ' + str(
                                            alphas) + '), solver = lsqr', X, y, solver='lsqr',
                                        covariance_estimator=GraphicalLassoCV(mode='lars', n_refinements=alphas),
                                        stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)
        run_algorithm_lda_configuration(metrics,
                                        'LDA: covariance_estimator = GraphicalLassoCV(assume_centered = True, n_refinements = ' + str(
                                            alphas) + '), solver = lsqr', X, y, solver='lsqr',
                                        covariance_estimator=GraphicalLassoCV(assume_centered=True,
                                                                              n_refinements=alphas), stratify=stratify,
                                        train_size=train_size, normalize_data=normalize_data, scaler=scaler)
        run_algorithm_lda_configuration(metrics,
                                        'LDA: covariance_estimator = GraphicalLassoCV(mode = lars, assume_centered = True, n_refinements = ' + str(
                                            alphas) + '), solver = lsqr', X, y, solver='lsqr',
                                        covariance_estimator=GraphicalLassoCV(mode='lars', assume_centered=True,
                                                                              n_refinements=alphas), stratify=stratify,
                                        train_size=train_size, normalize_data=normalize_data, scaler=scaler)

    for cv in range(2, 11, 1):
        run_algorithm_lda_configuration(metrics, 'LDA: covariance_estimator = GraphicalLassoCV(cv = ' + str(
            cv) + '), solver = lsqr',
                                        X, y, solver='lsqr', covariance_estimator=GraphicalLassoCV(cv=cv),
                                        stratify=stratify, train_size=train_size, normalize_data=normalize_data,
                                        scaler=scaler)
        run_algorithm_lda_configuration(metrics,
                                        'LDA: covariance_estimator = GraphicalLassoCV(mode = lars, cv = ' + str(
                                            cv) + '), solver = lsqr', X, y, solver='lsqr',
                                        covariance_estimator=GraphicalLassoCV(mode='lars', cv=cv), stratify=stratify,
                                        train_size=train_size, normalize_data=normalize_data, scaler=scaler)
        run_algorithm_lda_configuration(metrics,
                                        'LDA: covariance_estimator = GraphicalLassoCV(assume_centered = True, cv = ' + str(
                                            cv) + '), solver = lsqr', X, y, solver='lsqr',
                                        covariance_estimator=GraphicalLassoCV(assume_centered=True, cv=cv),
                                        stratify=stratify, train_size=train_size, normalize_data=normalize_data,
                                        scaler=scaler)
        run_algorithm_lda_configuration(metrics,
                                        'LDA: covariance_estimator = GraphicalLassoCV(mode = lars, assume_centered = True, cv = ' + str(
                                            cv) + '), solver = lsqr', X, y, solver='lsqr',
                                        covariance_estimator=GraphicalLassoCV(mode='lars', assume_centered=True, cv=cv),
                                        stratify=stratify, train_size=train_size, normalize_data=normalize_data,
                                        scaler=scaler)

    for tol in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        run_algorithm_lda_configuration(metrics, 'LDA: covariance_estimator = GraphicalLassoCV(tol = ' + str(
            tol) + '), solver = lsqr',
                                        X, y, solver='lsqr', covariance_estimator=GraphicalLassoCV(tol=tol),
                                        stratify=stratify, train_size=train_size, normalize_data=normalize_data,
                                        scaler=scaler)
        run_algorithm_lda_configuration(metrics,
                                        'LDA: covariance_estimator = GraphicalLassoCV(mode = lars, tol = ' + str(
                                            tol) + '), solver = lsqr', X, y, solver='lsqr',
                                        covariance_estimator=GraphicalLassoCV(mode='lars', tol=tol), stratify=stratify,
                                        train_size=train_size, normalize_data=normalize_data, scaler=scaler)
        run_algorithm_lda_configuration(metrics,
                                        'LDA: covariance_estimator = GraphicalLassoCV(assume_centered = True, tol = ' + str(
                                            tol) + '), solver = lsqr', X, y, solver='lsqr',
                                        covariance_estimator=GraphicalLassoCV(assume_centered=True, tol=tol),
                                        stratify=stratify, train_size=train_size, normalize_data=normalize_data,
                                        scaler=scaler)
        run_algorithm_lda_configuration(metrics,
                                        'LDA: covariance_estimator = GraphicalLassoCV(mode = lars, assume_centered = True, tol = ' + str(
                                            tol) + '), solver = lsqr', X, y, solver='lsqr',
                                        covariance_estimator=GraphicalLassoCV(mode='lars', assume_centered=True,
                                                                              tol=tol), stratify=stratify,
                                        train_size=train_size, normalize_data=normalize_data, scaler=scaler)
        run_algorithm_lda_configuration(metrics, 'LDA: covariance_estimator = GraphicalLassoCV(enet_tol = ' + str(
            tol) + '), solver = lsqr',
                                        X, y, solver='lsqr', covariance_estimator=GraphicalLassoCV(enet_tol=tol),
                                        stratify=stratify, train_size=train_size, normalize_data=normalize_data,
                                        scaler=scaler)
        run_algorithm_lda_configuration(metrics,
                                        'LDA: covariance_estimator = GraphicalLassoCV(mode = lars, enet_tol = ' + str(
                                            tol) + '), solver = lsqr', X, y, solver='lsqr',
                                        covariance_estimator=GraphicalLassoCV(mode='lars', enet_tol=tol),
                                        stratify=stratify, train_size=train_size, normalize_data=normalize_data,
                                        scaler=scaler)
        run_algorithm_lda_configuration(metrics,
                                        'LDA: covariance_estimator = GraphicalLassoCV(assume_centered = True, enet_tol = ' + str(
                                            tol) + '), solver = lsqr', X, y, solver='lsqr',
                                        covariance_estimator=GraphicalLassoCV(assume_centered=True, enet_tol=tol),
                                        stratify=stratify, train_size=train_size, normalize_data=normalize_data,
                                        scaler=scaler)
        run_algorithm_lda_configuration(metrics,
                                        'LDA: covariance_estimator = GraphicalLassoCV(mode = lars, assume_centered = True, enet_tol = ' + str(
                                            tol) + '), solver = lsqr', X, y, solver='lsqr',
                                        covariance_estimator=GraphicalLassoCV(mode='lars', assume_centered=True,
                                                                              enet_tol=tol), stratify=stratify,
                                        train_size=train_size, normalize_data=normalize_data, scaler=scaler)

    for max_iter in range(10, 100, 10):
        run_algorithm_lda_configuration(metrics, 'LDA: covariance_estimator = GraphicalLassoCV(max_iter = ' + str(
            max_iter) + '), solver = lsqr',
                                        X, y, solver='lsqr', covariance_estimator=GraphicalLassoCV(max_iter=max_iter),
                                        stratify=stratify, train_size=train_size, normalize_data=normalize_data,
                                        scaler=scaler)
        run_algorithm_lda_configuration(metrics,
                                        'LDA: covariance_estimator = GraphicalLassoCV(mode = lars, max_iter = ' + str(
                                            max_iter) + '), solver = lsqr', X, y, solver='lsqr',
                                        covariance_estimator=GraphicalLassoCV(mode='lars', max_iter=max_iter),
                                        stratify=stratify, train_size=train_size, normalize_data=normalize_data,
                                        scaler=scaler)
        run_algorithm_lda_configuration(metrics,
                                        'LDA: covariance_estimator = GraphicalLassoCV(assume_centered = True, max_iter = ' + str(
                                            max_iter) + '), solver = lsqr', X, y, solver='lsqr',
                                        covariance_estimator=GraphicalLassoCV(assume_centered=True, max_iter=max_iter),
                                        stratify=stratify, train_size=train_size, normalize_data=normalize_data,
                                        scaler=scaler)
        run_algorithm_lda_configuration(metrics,
                                        'LDA: covariance_estimator = GraphicalLassoCV(mode = lars, assume_centered = True, max_iter = ' + str(
                                            max_iter) + '), solver = lsqr', X, y, solver='lsqr',
                                        covariance_estimator=GraphicalLassoCV(mode='lars', assume_centered=True,
                                                                              max_iter=max_iter), stratify=stratify,
                                        train_size=train_size, normalize_data=normalize_data, scaler=scaler)

        run_algorithm_lda_configuration(metrics, 'LDA: covariance_estimator = GraphicalLassoCV(max_iter = ' + str(
            max_iter + 100) + '), solver = lsqr',
                                        X, y, solver='lsqr',
                                        covariance_estimator=GraphicalLassoCV(max_iter=max_iter + 100),
                                        stratify=stratify, train_size=train_size, normalize_data=normalize_data,
                                        scaler=scaler)
        run_algorithm_lda_configuration(metrics,
                                        'LDA: covariance_estimator = GraphicalLassoCV(mode = lars, max_iter = ' + str(
                                            max_iter + 100) + '), solver = lsqr', X, y, solver='lsqr',
                                        covariance_estimator=GraphicalLassoCV(mode='lars', max_iter=max_iter + 100),
                                        stratify=stratify, train_size=train_size, normalize_data=normalize_data,
                                        scaler=scaler)
        run_algorithm_lda_configuration(metrics,
                                        'LDA: covariance_estimator = GraphicalLassoCV(assume_centered = True, max_iter = ' + str(
                                            max_iter + 100) + '), solver = lsqr', X, y, solver='lsqr',
                                        covariance_estimator=GraphicalLassoCV(assume_centered=True,
                                                                              max_iter=max_iter + 100),
                                        stratify=stratify, train_size=train_size, normalize_data=normalize_data,
                                        scaler=scaler)
        run_algorithm_lda_configuration(metrics,
                                        'LDA: covariance_estimator = GraphicalLassoCV(mode = lars, assume_centered = True, max_iter = ' + str(
                                            max_iter + 100) + '), solver = lsqr', X, y, solver='lsqr',
                                        covariance_estimator=GraphicalLassoCV(mode='lars', assume_centered=True,
                                                                              max_iter=max_iter + 100),
                                        stratify=stratify, train_size=train_size,
                                        normalize_data=normalize_data, scaler=scaler)

    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_LDA)

    for tol in np.random.uniform(low=1e-7, high=0.02, size=(1000,)):
        for solver in ['lsqr', 'svd', 'eigen']:
            for shrinkage in ['auto', None]:
                for store_covariance in [True, False]:
                    for n_components in [0, 1, 2, None]:
                        try:
                            run_algorithm_lda_configuration(metrics, 'LDA: tol = ' + str(tol) + ', solver' + solver +
                                                            ', shrinkage= ' + str(
                                shrinkage) + ', store_covariance=' + str(
                                store_covariance)
                                                            + ', n_components' + str(
                                n_components) + ', covariance_estimator = None',
                                                            X, y, tol=tol, solver=solver,
                                                            shrinkage=shrinkage, store_covariance=store_covariance,
                                                            n_components=n_components, covariance_estimator=None,
                                                            stratify=stratify, train_size=train_size,
                                                            normalize_data=normalize_data, scaler=scaler)
                            run_algorithm_lda_configuration(metrics, 'LDA: tol = ' + str(tol) + ', solver' + solver +
                                                            ', shrinkage= ' + str(
                                shrinkage) + ', store_covariance=' + str(
                                store_covariance)
                                                            + ', n_components' + str(n_components) +
                                                            ', covariance_estimator = EllipticEnvelope(random_state=0)',
                                                            X, y, tol=tol, solver=solver,
                                                            shrinkage=shrinkage, store_covariance=store_covariance,
                                                            n_components=n_components,
                                                            covariance_estimator=EllipticEnvelope(random_state=0),
                                                            stratify=stratify, train_size=train_size,
                                                            normalize_data=normalize_data, scaler=scaler)
                            run_algorithm_lda_configuration(metrics, 'LDA: tol = ' + str(tol) + ', solver' + solver +
                                                            ', shrinkage= ' + str(
                                shrinkage) + ', store_covariance=' + str(
                                store_covariance)
                                                            + ', n_components' + str(n_components) +
                                                            ', covariance_estimator = EmpiricalCovariance()',
                                                            X, y, tol=tol, solver=solver,
                                                            shrinkage=shrinkage, store_covariance=store_covariance,
                                                            n_components=n_components,
                                                            covariance_estimator=EmpiricalCovariance(),
                                                            stratify=stratify, train_size=train_size,
                                                            normalize_data=normalize_data, scaler=scaler)
                            run_algorithm_lda_configuration(metrics, 'LDA: tol = ' + str(tol) + ', solver' + solver +
                                                            ', shrinkage= ' + str(
                                shrinkage) + ', store_covariance=' + str(
                                store_covariance)
                                                            + ', n_components' + str(n_components) +
                                                            ', covariance_estimator = GraphicalLassoCV()',
                                                            X, y, tol=tol, solver=solver,
                                                            shrinkage=shrinkage, store_covariance=store_covariance,
                                                            n_components=n_components,
                                                            covariance_estimator=GraphicalLassoCV(),
                                                            stratify=stratify, train_size=train_size,
                                                            normalize_data=normalize_data, scaler=scaler)
                        except Exception as err:
                            print(err)
                    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_LDA)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_LDA)
