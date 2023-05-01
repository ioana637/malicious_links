import os
import warnings
from itertools import chain
from multiprocessing import Manager, Pool

from data_pre import load_normalized_dataset, split_data_in_testing_training

warnings.filterwarnings("error")

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, RBF, Matern, RationalQuadratic, DotProduct

from utils import prediction, cal_metrics, appendMetricsTOCSV, convert_metrics_to_csv, listener_write_to_file


def create_label_for_GPC(kernel, n_restarts_optimizer, max_iter_predict, copy_X_train, multi_class):
    return "GPC, kernel=" + str(kernel) + ", n_restarts_optimizer=" + str(
        n_restarts_optimizer) + ", max_iter_predict=" + \
           str(max_iter_predict) + ", copy_X_train=" + str(copy_X_train) + ", multi_class=" + str(multi_class)


def run_algorithm_gaussian_process_configuration_parallel(X, y, q_metrics,
                                                          kernel=None,
                                                          n_restarts_optimizer=0,
                                                          max_iter_predict=100,
                                                          copy_X_train=True,
                                                          multi_class='one_vs_rest',
                                                          stratify=False, train_size=0.8,
                                                          ):
    X_train, X_test, y_train, y_test = split_data_in_testing_training(X, y, stratify, train_size)

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
        label = create_label_for_GPC(kernel, n_restarts_optimizer, max_iter_predict, copy_X_train, multi_class)
        precision, recall, f1, roc_auc = cal_metrics(y_test, y_pred, y_pred_probabilities, label, classifier)
        string_results_for_queue = convert_metrics_to_csv(',', label,
                                                          kernel, n_restarts_optimizer, max_iter_predict, copy_X_train,
                                                          multi_class, precision, recall, f1, roc_auc)
        q_metrics.put(string_results_for_queue)
    except Exception as er:
        print(er)
    except RuntimeWarning as warn:
        print(warn)


def run_algorithm_gaussian_process_configuration(metrics, label, X, y,
                                                 kernel=None,
                                                 n_restarts_optimizer=0,
                                                 max_iter_predict=100,
                                                 copy_X_train=True,
                                                 multi_class='one_vs_rest',
                                                 stratify=False, train_size=0.8,
                                                 ):
    X_train, X_test, y_train, y_test = split_data_in_testing_training(X, y, stratify, train_size)

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


def run_algorithm_gpc(filename='', path='', stratify=False, train_size=0.8,
                      normalize_data=False, scaler='min-max'):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_GPC()

    # full_path_filename = '/content/drive/MyDrive/code/' + filename
    # path_to_script = os.path.dirname(os.path.abspath(__file__))
    my_filename = os.path.join(path, 'results/gpc', filename)

    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_GPC, header=True)

    # # default algorithm
    # run_algorithm_gaussian_process_configuration(metrics, 'Gaussian Process Classifier  - default params', X, y,
    #                                              stratify=stratify, train_size=train_size,
    #                                              )
    # metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_GPC)
    #
    # # CALIBRATING multi_class
    # run_algorithm_gaussian_process_configuration(metrics, 'Gaussian Process Classifier  - multi_class = one vs one', X,
    #                                              y, multi_class='one_vs_one', stratify=stratify,
    #                                              train_size=train_size)
    # # CALIBRATING copy_X_train
    # run_algorithm_gaussian_process_configuration(metrics, 'Gaussian Process Classifier  - copy_X_train = False', X, y,
    #                                              copy_X_train=False, stratify=stratify, train_size=train_size,
    #                                              )
    # run_algorithm_gaussian_process_configuration(metrics,
    #                                              'Gaussian Process Classifier  - multi_class = one vs one, copy_X_train = False',
    #                                              X, y, copy_X_train=False, multi_class='one_vs_one',
    #                                              stratify=stratify, train_size=train_size,
    #                                              )
    #
    # # CALIBRATING kernel
    # run_algorithm_gaussian_process_configuration(metrics, 'Gaussian Process Classifier  - kernel = Constant Kernel', X,
    #                                              y, kernel=ConstantKernel(constant_value_bounds=[1e-10, 1e10]),
    #                                              stratify=stratify, train_size=train_size,
    #                                              )
    # run_algorithm_gaussian_process_configuration(metrics, 'Gaussian Process Classifier  - kernel = White Kernel', X, y,
    #                                              kernel=WhiteKernel(noise_level_bounds=[1e-10, 1e10]),
    #                                              stratify=stratify, train_size=train_size,
    #                                              )
    # run_algorithm_gaussian_process_configuration(metrics, 'Gaussian Process Classifier  - kernel = RBF Kernel', X, y,
    #                                              kernel=RBF(length_scale_bounds=[1e-10, 1e10]), stratify=stratify,
    #                                              train_size=train_size)
    #
    # run_algorithm_gaussian_process_configuration(metrics,
    #                                              'Gaussian Process Classifier  - kernel = Matern Kernel, nu = 0.5 ', X,
    #                                              y, kernel=Matern(nu=0.5, length_scale_bounds=[1e-10, 1e10]),
    #                                              stratify=stratify, train_size=train_size)
    # run_algorithm_gaussian_process_configuration(metrics,
    #                                              'Gaussian Process Classifier  - kernel = Matern Kernel, nu = 1.5', X,
    #                                              y, kernel=Matern(nu=1.5, length_scale_bounds=[1e-10, 1e10]),
    #                                              stratify=stratify, train_size=train_size)
    # run_algorithm_gaussian_process_configuration(metrics,
    #                                              'Gaussian Process Classifier  - kernel = Matern Kernel, nu = 2.5', X,
    #                                              y, kernel=Matern(nu=2.5, length_scale_bounds=[1e-10, 1e10]),
    #                                              stratify=stratify, train_size=train_size)
    #
    # run_algorithm_gaussian_process_configuration(metrics,
    #                                              'Gaussian Process Classifier  - kernel = RationalQuadratic Kernel', X,
    #                                              y, kernel=RationalQuadratic(alpha_bounds=[1e-10, 1e10],
    #                                                                          length_scale_bounds=[1e-10, 1e10]),
    #                                              stratify=stratify, train_size=train_size)
    # # run_algorithm_gaussian_process_configuration(metrics, 'Gaussian Process Classifier  - kernel = ExpSineSquared Kernel', X, y, kernel = ExpSineSquared(periodicity_bounds = [1e-10, 1e10], length_scale_bounds = [1e-10, 1e10]), stratify = stratify, train_size = train_size)
    # run_algorithm_gaussian_process_configuration(metrics, 'Gaussian Process Classifier  - kernel = DotProduct Kernel',
    #                                              X, y, kernel=DotProduct(sigma_0_bounds=[1e-10, 1e10]),
    #                                              stratify=stratify, train_size=train_size)
    #
    # # CALIBRATING n_restarts_optimizer
    # for n_restarts_optimizer in chain(range(1, 20, 1), range(20, 50, 2), range(50, 100, 5), range(100, 200, 10),
    #                                   range(200, 500, 25), range(500, 1000, 50)):
    #     run_algorithm_gaussian_process_configuration(metrics,
    #                                                  'Gaussian Process Classifier  - n_restarts_optimizer = ' + str(
    #                                                      n_restarts_optimizer), X, y,
    #                                                  n_restarts_optimizer=n_restarts_optimizer, stratify=stratify,
    #                                                  train_size=train_size)
    #     # CALIBRATING multi_class
    #     run_algorithm_gaussian_process_configuration(metrics,
    #                                                  'Gaussian Process Classifier  - multi_class = one vs one, n_restarts_optimizer = ' + str(
    #                                                      n_restarts_optimizer), X, y,
    #                                                  n_restarts_optimizer=n_restarts_optimizer,
    #                                                  multi_class='one_vs_one', stratify=stratify, train_size=train_size,
    #                                                  )
    #     # CALIBRATING copy_X_train
    #     run_algorithm_gaussian_process_configuration(metrics,
    #                                                  'Gaussian Process Classifier  - copy_X_train = False, n_restarts_optimizer = ' + str(
    #                                                      n_restarts_optimizer), X, y,
    #                                                  n_restarts_optimizer=n_restarts_optimizer, copy_X_train=False,
    #                                                  stratify=stratify, train_size=train_size,
    #                                                  )
    #     run_algorithm_gaussian_process_configuration(metrics,
    #                                                  'Gaussian Process Classifier  - multi_class = one vs one, copy_X_train = False, n_restarts_optimizer = ' + str(
    #                                                      n_restarts_optimizer), X, y,
    #                                                  n_restarts_optimizer=n_restarts_optimizer, copy_X_train=False,
    #                                                  multi_class='one_vs_one', stratify=stratify, train_size=train_size,
    #                                                  )
    #
    # # CALIBRATING max_iter_predict
    # for max_iter_predict in chain(range(1, 40, 5), range(40, 70, 2), range(70, 100, 1), range(100, 130, 1),
    #                               range(130, 160, 2), range(160, 200, 5), range(200, 300, 10), range(300, 500, 25),
    #                               range(500, 1000, 50)):
    #     run_algorithm_gaussian_process_configuration(metrics,
    #                                                  'Gaussian Process Classifier  - max_iter_predict = ' + str(
    #                                                      max_iter_predict), X, y, max_iter_predict=max_iter_predict,
    #                                                  stratify=stratify, train_size=train_size
    #                                                  )
    #     # CALIBRATING multi_class
    #     run_algorithm_gaussian_process_configuration(metrics,
    #                                                  'Gaussian Process Classifier  - multi_class = one vs one, max_iter_predict = ' + str(
    #                                                      max_iter_predict), X, y, max_iter_predict=max_iter_predict,
    #                                                  multi_class='one_vs_one', stratify=stratify, train_size=train_size,
    #                                                  )
    #     # CALIBRATING copy_X_train
    #     run_algorithm_gaussian_process_configuration(metrics,
    #                                                  'Gaussian Process Classifier  - copy_X_train = False, max_iter_predict = ' + str(
    #                                                      max_iter_predict), X, y, max_iter_predict=max_iter_predict,
    #                                                  copy_X_train=False, stratify=stratify, train_size=train_size,
    #                                                 )
    #     run_algorithm_gaussian_process_configuration(metrics,
    #                                                  'Gaussian Process Classifier  - multi_class = one vs one, copy_X_train = False, max_iter_predict = ' + str(
    #                                                      max_iter_predict), X, y, max_iter_predict=max_iter_predict,
    #                                                  copy_X_train=False, multi_class='one_vs_one', stratify=stratify,
    #                                                  train_size=train_size)

    # metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_GPC)
    for max_iter_predict in chain(range(30, 150, 1), range(795, 805, 1)):
        for n_restarts_optimizer in chain(range(0, 26, 1), range(90, 110, 1), range(190, 210, 1),
                                          range(365, 385, 1), range(440, 485, 1)):
            for copy_X_train in [True, False]:
                for multiclass in ['one_vs_rest', 'one_vs_one']:
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
                                                                 train_size=train_size)
                    for sigma_0 in chain([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
                                          0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                                          0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                          1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9], range(2, 20, 1),
                                         range(20, 50, 2), range(50, 100, 5), range(100, 200, 10),
                                         range(200, 400, 25), range(400, 1000, 50), range(1000, 10000, 100),
                                         range(10000, 90000, 500)):
                        run_algorithm_gaussian_process_configuration(metrics,
                                                                     'Gaussian Process Classifier  - multi_class = ' +
                                                                     multiclass + ', copy_X_train = ' + str(
                                                                         copy_X_train) +
                                                                     ', max_iter_predict = ' + str(
                                                                         max_iter_predict) + ', n_restarts_optimizer=' + str(
                                                                         n_restarts_optimizer) +
                                                                     ', kernel = DotProductKernel(sigma_0=' + str(
                                                                         sigma_0) + ")", X, y,
                                                                     max_iter_predict=max_iter_predict,
                                                                     n_restarts_optimizer=n_restarts_optimizer,
                                                                     kernel=DotProduct(sigma_0=sigma_0),
                                                                     copy_X_train=copy_X_train, multi_class=multiclass,
                                                                     stratify=stratify,
                                                                     train_size=train_size)
                    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_GPC)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_GPC)


def run_algorithm_gpc_parallel(filename='', path='', stratify=False, train_size=0.8,
                               normalize_data=False, scaler='min-max', no_threads=8):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_GPC()
    my_filename = os.path.join(path, 'results/gpc', filename)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_GPC, header=True)

    max_iter_predict_list = list(chain(range(30, 150, 4), range(795, 805, 2)))
    n_restarts_optimizer_list = list(chain(range(0, 26, 3), [90, 100, 110], [190, 200, 210],
                                           [360, 370, 380, 390], [440, 450, 460, 470, 480, 490]))
    copy_X_train_list = [True, False]
    multiclass_list = ['one_vs_rest', 'one_vs_one']
    sigma_0_list = list(chain([0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 1.5],
                              range(2, 15, 4), range(15, 30, 5), range(30, 60, 10), range(60, 150, 20),
                              range(150, 400, 50), range(400, 1000, 200), range(1000, 5000, 500),
                              range(5000, 10000, 1000), range(10000, 90000, 5000)))
    # 213500

    with Manager() as manager:
        q_metrics = manager.Queue()
        jobs = []

        with Pool(no_threads) as pool:
            watcher = pool.apply_async(listener_write_to_file, (q_metrics, my_filename))
            for max_iter_predict in max_iter_predict_list:
                for n_restarts_optimizer in n_restarts_optimizer_list:
                    for copy_X_train in copy_X_train_list:
                        for multiclass in multiclass_list:
                            job = pool.apply_async(run_algorithm_gaussian_process_configuration_parallel,
                                                   (X, y, q_metrics, None, n_restarts_optimizer, max_iter_predict,
                                                    copy_X_train, multiclass, stratify, train_size))
                            jobs.append(job)
                            for sigma_0 in sigma_0_list:
                                job = pool.apply_async(run_algorithm_gaussian_process_configuration_parallel,
                                                       (X, y, q_metrics,
                                                        DotProduct(sigma_0=sigma_0, sigma_0_bounds=(1e-5, 1e6)),
                                                        n_restarts_optimizer, max_iter_predict, copy_X_train,
                                                        multiclass, stratify, train_size))
                                jobs.append(job)
            # print(jobs)
            # collect results from the workers through the pool result queue
            for job in jobs:
                job.get()
            # now we are done, kill the listener
            q_metrics.put('kill')
            pool.close()
            pool.join()
