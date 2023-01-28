import os
from itertools import chain
import warnings
warnings.filterwarnings("error")

import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, RBF, Matern, RationalQuadratic, DotProduct

from utils import split_data, prediction, cal_metrics, appendMetricsTOCSV


def run_algorithm_gaussian_process_configuration(metrics, label, X, y,
                                                 kernel=None,
                                                 n_restarts_optimizer=0,
                                                 max_iter_predict=100,
                                                 copy_X_train=True,
                                                 multi_class='one_vs_rest',
                                                 stratify=False, train_size=0.8,
                                                 normalize_data=False, scaler='min-max'
                                                 ):
    X_train, X_test, y_train, y_test = split_data(X, y, normalize_data=normalize_data, stratify=stratify,
                                                  train_size=train_size, scaler=scaler)
    try:
    # Creating the classifier object
        classifier = GaussianProcessClassifier(kernel=kernel, copy_X_train=copy_X_train,
                                           multi_class=multi_class, max_iter_predict=max_iter_predict,
                                           n_restarts_optimizer=n_restarts_optimizer, n_jobs=-1
                                           )
    # Performing training
        classifier.fit(X_train, y_train)

    # Make predictions
        y_pred, y_pred_probabilities = prediction(X_test, classifier)

    # Compute metrics
        precision, recall, f1, roc_auc = cal_metrics(y_test, y_pred, y_pred_probabilities, label)
        metrics['label'].append(label)

        metrics['kernel'].append(kernel)
        metrics['n_restarts_optimizer'].append(n_restarts_optimizer)
        metrics['max_iter_predict'].append(max_iter_predict)
        metrics['copy_X_train'].append(copy_X_train)
        metrics['multi_class'].append(multi_class)

        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1_score'].append(f1)
        metrics['roc_auc'].append(roc_auc)
    except Exception as er:
        pass
    except RuntimeWarning as warn:
        pass


def init_metrics_for_GPC():
    return {'label': [],
            'kernel': [],
            'n_restarts_optimizer': [],
            'max_iter_predict': [],
            'copy_X_train': [],
            'multi_class': [],
            'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': []
            }


def run_algorithm_gpc(df, filename='', stratify=False, train_size=0.8, normalize_data=False, scaler='min-max'):
    y = df['label'].copy()
    X = df.drop('label', axis=1).copy()
    metrics = init_metrics_for_GPC()

    path_to_script = os.path.dirname(os.path.abspath(__file__))
    my_filename = os.path.join(path_to_script, 'gpc', filename)

    # default algorithm
    run_algorithm_gaussian_process_configuration(metrics, 'Gaussian Process Classifier  - default params', X, y,
                                                 stratify=stratify, train_size=train_size,
                                                 normalize_data=normalize_data, scaler=scaler)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_GPC, header=True)

    # # CALIBRATING multi_class
    # run_algorithm_gaussian_process_configuration(metrics, 'Gaussian Process Classifier  - multi_class = one vs one', X,
    #                                              y, multi_class='one_vs_one', stratify=stratify,
    #                                              train_size=train_size, normalize_data=normalize_data, scaler=scaler)
    #
    # # CALIBRATING copy_X_train
    # run_algorithm_gaussian_process_configuration(metrics, 'Gaussian Process Classifier  - copy_X_train = False', X, y,
    #                                              copy_X_train=False, stratify=stratify, train_size=train_size,
    #                                              normalize_data=normalize_data, scaler=scaler)
    # run_algorithm_gaussian_process_configuration(metrics,
    #                                              'Gaussian Process Classifier  - multi_class = one vs one, copy_X_train = False',
    #                                              X, y, copy_X_train=False, multi_class='one_vs_one',
    #                                              stratify=stratify, train_size=train_size,
    #                                              normalize_data=normalize_data, scaler=scaler)
    #
    # # CALIBRATING kernel
    # run_algorithm_gaussian_process_configuration(metrics, 'Gaussian Process Classifier  - kernel = Constant Kernel', X,
    #                                              y, kernel=ConstantKernel(constant_value_bounds=[1e-10, 1e10]),
    #                                              stratify=stratify, train_size=train_size,
    #                                              normalize_data=normalize_data, scaler=scaler)
    # run_algorithm_gaussian_process_configuration(metrics, 'Gaussian Process Classifier  - kernel = White Kernel', X, y,
    #                                              kernel=WhiteKernel(noise_level_bounds=[1e-10, 1e10]),
    #                                              stratify=stratify, train_size=train_size,
    #                                              normalize_data=normalize_data, scaler=scaler)
    # run_algorithm_gaussian_process_configuration(metrics, 'Gaussian Process Classifier  - kernel = RBF Kernel', X, y,
    #                                              kernel=RBF(length_scale_bounds=[1e-10, 1e10]), stratify=stratify,
    #                                              train_size=train_size,
    #                                              normalize_data=normalize_data, scaler=scaler)
    #
    # run_algorithm_gaussian_process_configuration(metrics,
    #                                              'Gaussian Process Classifier  - kernel = Matern Kernel, nu = 0.5 ', X,
    #                                              y, kernel=Matern(nu=0.5, length_scale_bounds=[1e-10, 1e10]),
    #                                              stratify=stratify, train_size=train_size,
    #                                              normalize_data=normalize_data, scaler=scaler)
    # run_algorithm_gaussian_process_configuration(metrics,
    #                                              'Gaussian Process Classifier  - kernel = Matern Kernel, nu = 1.5', X,
    #                                              y, kernel=Matern(nu=1.5, length_scale_bounds=[1e-10, 1e10]),
    #                                              stratify=stratify, train_size=train_size,
    #                                              normalize_data=normalize_data, scaler=scaler)
    # run_algorithm_gaussian_process_configuration(metrics,
    #                                              'Gaussian Process Classifier  - kernel = Matern Kernel, nu = 2.5', X,
    #                                              y, kernel=Matern(nu=2.5, length_scale_bounds=[1e-10, 1e10]),
    #                                              stratify=stratify, train_size=train_size,
    #                                              normalize_data=normalize_data, scaler=scaler)
    #
    # run_algorithm_gaussian_process_configuration(metrics,
    #                                              'Gaussian Process Classifier  - kernel = RationalQuadratic Kernel', X,
    #                                              y, kernel=RationalQuadratic(alpha_bounds=[1e-10, 1e10],
    #                                                                          length_scale_bounds=[1e-10, 1e10]),
    #                                              stratify=stratify, train_size=train_size,
    #                                              normalize_data=normalize_data, scaler=scaler)
    # # run_algorithm_gaussian_process_configuration(metrics, 'Gaussian Process Classifier  - kernel = ExpSineSquared Kernel', X, y, kernel = ExpSineSquared(periodicity_bounds = [1e-10, 1e10], length_scale_bounds = [1e-10, 1e10]), stratify = stratify, train_size = train_size)
    # run_algorithm_gaussian_process_configuration(metrics, 'Gaussian Process Classifier  - kernel = DotProduct Kernel',
    #                                              X, y, kernel=DotProduct(sigma_0_bounds=[1e-10, 1e10]),
    #                                              stratify=stratify, train_size=train_size,
    #                                              normalize_data=normalize_data, scaler=scaler)
    #
    # # CALIBRATING n_restarts_optimizer
    # for n_restarts_optimizer in chain(range(1, 20, 1), range(20, 50, 2), range(50, 100, 5), range(100, 200, 10),
    #                                   range(200, 500, 25), range(500, 1000, 50)):
    #     run_algorithm_gaussian_process_configuration(metrics,
    #                                                  'Gaussian Process Classifier  - n_restarts_optimizer = ' + str(
    #                                                      n_restarts_optimizer), X, y,
    #                                                  n_restarts_optimizer=n_restarts_optimizer, stratify=stratify,
    #                                                  train_size=train_size, normalize_data=normalize_data,
    #                                                  scaler=scaler)
    #     # CALIBRATING multi_class
    #     run_algorithm_gaussian_process_configuration(metrics,
    #                                                  'Gaussian Process Classifier  - multi_class = one vs one, n_restarts_optimizer = ' + str(
    #                                                      n_restarts_optimizer), X, y,
    #                                                  n_restarts_optimizer=n_restarts_optimizer,
    #                                                  multi_class='one_vs_one', stratify=stratify, train_size=train_size,
    #                                                  normalize_data=normalize_data, scaler=scaler)
    #     # CALIBRATING copy_X_train
    #     run_algorithm_gaussian_process_configuration(metrics,
    #                                                  'Gaussian Process Classifier  - copy_X_train = False, n_restarts_optimizer = ' + str(
    #                                                      n_restarts_optimizer), X, y,
    #                                                  n_restarts_optimizer=n_restarts_optimizer, copy_X_train=False,
    #                                                  stratify=stratify, train_size=train_size,
    #                                                  normalize_data=normalize_data, scaler=scaler)
    #     run_algorithm_gaussian_process_configuration(metrics,
    #                                                  'Gaussian Process Classifier  - multi_class = one vs one, copy_X_train = False, n_restarts_optimizer = ' + str(
    #                                                      n_restarts_optimizer), X, y,
    #                                                  n_restarts_optimizer=n_restarts_optimizer, copy_X_train=False,
    #                                                  multi_class='one_vs_one', stratify=stratify, train_size=train_size,
    #                                                  normalize_data=normalize_data, scaler=scaler)
    #
    # # CALIBRATING max_iter_predict
    # for max_iter_predict in chain(range(1, 40, 5), range(40, 70, 2), range(70, 100, 1), range(100, 130, 1),
    #                               range(130, 160, 2), range(160, 200, 5), range(200, 300, 10), range(300, 500, 25),
    #                               range(500, 1000, 50)):
    #     run_algorithm_gaussian_process_configuration(metrics,
    #                                                  'Gaussian Process Classifier  - max_iter_predict = ' + str(
    #                                                      max_iter_predict), X, y, max_iter_predict=max_iter_predict,
    #                                                  stratify=stratify, train_size=train_size,
    #                                                  normalize_data=normalize_data, scaler=scaler)
    #     # CALIBRATING multi_class
    #     run_algorithm_gaussian_process_configuration(metrics,
    #                                                  'Gaussian Process Classifier  - multi_class = one vs one, max_iter_predict = ' + str(
    #                                                      max_iter_predict), X, y, max_iter_predict=max_iter_predict,
    #                                                  multi_class='one_vs_one', stratify=stratify, train_size=train_size,
    #                                                  normalize_data=normalize_data, scaler=scaler)
    #     # CALIBRATING copy_X_train
    #     run_algorithm_gaussian_process_configuration(metrics,
    #                                                  'Gaussian Process Classifier  - copy_X_train = False, max_iter_predict = ' + str(
    #                                                      max_iter_predict), X, y, max_iter_predict=max_iter_predict,
    #                                                  copy_X_train=False, stratify=stratify, train_size=train_size,
    #                                                  normalize_data=normalize_data, scaler=scaler)
    #     run_algorithm_gaussian_process_configuration(metrics,
    #                                                  'Gaussian Process Classifier  - multi_class = one vs one, copy_X_train = False, max_iter_predict = ' + str(
    #                                                      max_iter_predict), X, y, max_iter_predict=max_iter_predict,
    #                                                  copy_X_train=False, multi_class='one_vs_one', stratify=stratify,
    #                                                  train_size=train_size, normalize_data=normalize_data,
    #                                                  scaler=scaler)

    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_GPC)
    for max_iter_predict in chain(range(945, 955, 2), range(895, 905, 2), range(845, 855, 2), range(795, 805, 2),
                                  range(745, 755, 2), range(94, 104, 2), range(53, 63, 2), range(11, 21, 2),
                                  range(119, 129, 2), range(78, 93, 2), range(68, 78, 2), range(445, 455, 2),
                                  range(185, 195, 2)):
        # 60+8=68
        for n_restarts_optimizer in chain(range(0, 5), range(9, 19, 2), range(115, 125, 2), range(545, 555, 2),
                                          range(345, 355, 2)):
            # 5+20=25
            for copy_X_train in [True, False]:
                for multiclass in ['one_vs_rest', 'one_vs_one']:
                    # 24
                    for nu in chain(np.random.uniform(low=0.0001, high=3.0, size=(20,))):
                        run_algorithm_gaussian_process_configuration(metrics,
                                                                     'Gaussian Process Classifier  - multi_class = ' +
                                                                     multiclass + ', copy_X_train = ' + str(
                                                                         copy_X_train) +
                                                                     ', max_iter_predict = ' + str(
                                                                         max_iter_predict) + ', n_restarts_optimizer=' + str(
                                                                         n_restarts_optimizer) +
                                                                     ', kernel = MaternKernel (nu=' + str(nu) + ')', X,
                                                                     y,
                                                                     max_iter_predict=max_iter_predict,
                                                                     n_restarts_optimizer=n_restarts_optimizer,
                                                                     kernel=Matern(nu=nu),
                                                                     copy_X_train=copy_X_train, multi_class=multiclass,
                                                                     stratify=stratify,
                                                                     train_size=train_size,
                                                                     normalize_data=normalize_data, scaler=scaler)
                    run_algorithm_gaussian_process_configuration(metrics,
                                                                 'Gaussian Process Classifier  - multi_class = ' +
                                                                 multiclass + ', copy_X_train = ' + str(copy_X_train) +
                                                                 ', max_iter_predict = ' + str(
                                                                     max_iter_predict) + ', n_restarts_optimizer=' + str(
                                                                     n_restarts_optimizer) +
                                                                 ', kernel = None', X, y,
                                                                 max_iter_predict=max_iter_predict,
                                                                 n_restarts_optimizer=n_restarts_optimizer,
                                                                 kernel=None,
                                                                 copy_X_train=copy_X_train, multi_class=multiclass,
                                                                 stratify=stratify,
                                                                 train_size=train_size, normalize_data=normalize_data,
                                                                 scaler=scaler)
                    run_algorithm_gaussian_process_configuration(metrics,
                                                                 'Gaussian Process Classifier  - multi_class = ' +
                                                                 multiclass + ', copy_X_train = ' + str(copy_X_train) +
                                                                 ', max_iter_predict = ' + str(
                                                                     max_iter_predict) + ', n_restarts_optimizer=' + str(
                                                                     n_restarts_optimizer) +
                                                                 ', kernel = DotProductKernel', X, y,
                                                                 max_iter_predict=max_iter_predict,
                                                                 n_restarts_optimizer=n_restarts_optimizer,
                                                                 kernel=DotProduct(),
                                                                 copy_X_train=copy_X_train, multi_class=multiclass,
                                                                 stratify=stratify,
                                                                 train_size=train_size, normalize_data=normalize_data,
                                                                 scaler=scaler)
                    run_algorithm_gaussian_process_configuration(metrics,
                                                                 'Gaussian Process Classifier  - multi_class = ' +
                                                                 multiclass + ', copy_X_train = ' + str(copy_X_train) +
                                                                 ', max_iter_predict = ' + str(
                                                                     max_iter_predict) + ', n_restarts_optimizer=' + str(
                                                                     n_restarts_optimizer) +
                                                                 ', kernel = RationalQuadraticKernel', X, y,
                                                                 max_iter_predict=max_iter_predict,
                                                                 n_restarts_optimizer=n_restarts_optimizer,
                                                                 kernel=RationalQuadratic(),
                                                                 copy_X_train=copy_X_train, multi_class=multiclass,
                                                                 stratify=stratify,
                                                                 train_size=train_size, normalize_data=normalize_data,
                                                                 scaler=scaler)
                    run_algorithm_gaussian_process_configuration(metrics,
                                                                 'Gaussian Process Classifier  - multi_class = ' +
                                                                 multiclass + ', copy_X_train = ' + str(copy_X_train) +
                                                                 ', max_iter_predict = ' + str(
                                                                     max_iter_predict) + ', n_restarts_optimizer=' + str(
                                                                     n_restarts_optimizer) +
                                                                 ', kernel = RBF kernel', X, y,
                                                                 max_iter_predict=max_iter_predict,
                                                                 n_restarts_optimizer=n_restarts_optimizer,
                                                                 kernel=RBF(),
                                                                 copy_X_train=copy_X_train, multi_class=multiclass,
                                                                 stratify=stratify,
                                                                 train_size=train_size, normalize_data=normalize_data,
                                                                 scaler=scaler)
                    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_GPC)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_GPC)
