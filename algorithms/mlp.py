import os
from multiprocessing import Pool, Manager

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.utils._testing import ignore_warnings

from data_post import compute_average_metric
from data_pre import load_normalized_dataset, split_data_in_testing_training
from utils import prediction, cal_metrics, appendMetricsTOCSV, convert_metrics_to_csv, listener_write_to_file


def run_top_20_MLP_configs(filename='', path='', stratify=False, train_size=0.8,
                           normalize_data=True, scaler='min-max', n_rep=100):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_MLP()

    # full_path_filename = '/content/drive/MyDrive/code/' + filename
    # path_to_script = os.path.dirname(os.path.abspath(__file__))
    my_filename = os.path.join(path, 'results/mlp', filename)

    # TODO top 20 best configs

    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.groupby(['label'], as_index=False).agg({'precision': 'mean', 'recall': 'mean',
                                                                    'f1_score': 'mean', 'roc_auc': 'mean',

                                                                    'max_iter': 'first', 'hidden_layer_sizes': 'first',
                                                                    'random_state': 'first', 'batch_size': 'first',
                                                                    'activation': 'first', 'solver': 'first',
                                                                    'alpha': 'first',
                                                                    'learning_rate': 'first',
                                                                    'learning_rate_init': 'first',
                                                                    'power_t': 'first', 'shuffle': 'first',
                                                                    'momentum': 'first', 'nesterovs_momentum': 'first',
                                                                    'early_stopping': 'first',
                                                                    'validation_fraction': 'first',
                                                                    'beta_1': 'first', 'beta_2': 'first',
                                                                    'epsilon': 'first', 'n_iter_no_change': 'first',
                                                                    'max_fun': 'first'
                                                                    })
    metrics_df = compute_average_metric(metrics_df)
    metrics_df.sort_values(by=['average_metric'], ascending=False, inplace=True)
    metrics = appendMetricsTOCSV(my_filename, metrics_df, init_metrics_for_MLP, header=True)

@ignore_warnings(category=ConvergenceWarning)
def run_algorithm_MLP_configuration(metrics, label, X, y,
                                    max_iter=100, hidden_layer_sizes=[35, 100, 100],
                                    random_state=123, batch_size=1, activation='relu',
                                    solver='adam', alpha=0.0001, learning_rate='constant',
                                    learning_rate_init=0.001, power_t=0.5, shuffle=True,
                                    momentum=0.9, nesterovs_momentum=True, early_stopping=False,
                                    validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                                    epsilon=1e-8, n_iter_no_change=10, max_fun=15000,
                                    train_size=0.8, stratify=False):
    X_train, X_test, y_train, y_test = split_data_in_testing_training(X, y, stratify, train_size)

    try:
        # Creating the classifier object
        classifier = MLPClassifier(max_iter=max_iter, hidden_layer_sizes=hidden_layer_sizes, random_state=random_state,
                                   batch_size=batch_size,
                                   activation=activation, solver=solver, alpha=alpha, learning_rate=learning_rate,
                                   learning_rate_init=learning_rate_init, power_t=power_t, shuffle=shuffle,
                                   momentum=momentum, nesterovs_momentum=nesterovs_momentum,
                                   early_stopping=early_stopping,
                                   validation_fraction=validation_fraction, beta_1=beta_1, beta_2=beta_2,
                                   epsilon=epsilon, n_iter_no_change=n_iter_no_change, max_fun=max_fun)

        # Performing training
        classifier.fit(X_train, y_train)

        # Make predictions
        y_pred, y_pred_probabilities = prediction(X_test, classifier)

        # Compute metrics
        precision, recall, f1, roc_auc = cal_metrics(y_test, y_pred, y_pred_probabilities, label, classifier)
        metrics['label'].append(label)
        metrics['max_iter'].append(max_iter)
        metrics['hidden_layer_sizes'].append(hidden_layer_sizes)
        metrics['random_state'].append(random_state)
        metrics['batch_size'].append(batch_size)
        metrics['activation'].append(activation)
        metrics['solver'].append(solver)
        metrics['alpha'].append(alpha)
        metrics['learning_rate'].append(learning_rate)
        metrics['learning_rate_init'].append(learning_rate_init)
        metrics['power_t'].append(power_t)
        metrics['shuffle'].append(shuffle)
        metrics['momentum'].append(momentum)
        metrics['nesterovs_momentum'].append(nesterovs_momentum)
        metrics['early_stopping'].append(early_stopping)
        metrics['validation_fraction'].append(validation_fraction)
        metrics['beta_1'].append(beta_1)
        metrics['beta_2'].append(beta_2)
        metrics['epsilon'].append(epsilon)
        metrics['n_iter_no_change'].append(n_iter_no_change)
        metrics['max_fun'].append(max_fun)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1_score'].append(f1)
        metrics['roc_auc'].append(roc_auc)
    except Exception as err:
        pass
        # print(err)
    except RuntimeWarning as warn:
        # print(warn)
        pass


def init_metrics_for_MLP():
    return {'label': [], 'max_iter': [], 'hidden_layer_sizes': [], 'random_state': [], 'batch_size': [],
            'activation': [], 'solver': [], 'alpha': [],
            'learning_rate': [], 'learning_rate_init': [], 'power_t': [],
            'shuffle': [], 'momentum': [], 'nesterovs_momentum': [],
            'early_stopping': [], 'validation_fraction': [], 'beta_1': [], 'beta_2': [],
            'epsilon': [], 'n_iter_no_change': [], 'max_fun': [],
            'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': []
            }


def run_algorithm_MLP_parallel(filename='', path='', stratify=False, train_size=0.8,
                               normalize_data=False, scaler='min-max', no_threads=8):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_MLP()
    my_filename = os.path.join(path, 'results/mlp', filename)
    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_MLP, header=True)
    max_iter_list = range(100, 200, 25)
    i_list = range(20, 150, 20)  # au fost cu range(10, 150, 10)
    j_list = range(20, 150, 20)
    k_list = range(20, 150, 20)
    n_iter_no_change_list = range(5, 20, 2)
    epsilon_list = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-14, 1e-15, 1e-16, 1e-17, 1e-18, 1e-19, 1e-20]
    learning_rate_init_list = [0.0005, 0.00055, 0.0006, 0.00065, 0.0007, 0.00075, 0.0008, 0.00085, 0.0009, 0.00095,
                               0.001, 0.0011, 0.0012, 0.0013, 0.0014, 0.0015, 0.0016, 0.00165, 0.0017, 0.00175, 0.0018,
                               0.00185, 0.0019, 0.00195, 0.002]
    solver_list = ['adam', 'sgd']
    activation_list = ['relu', 'identity', 'logistic', 'tanh']
    # count = len(max_iter_list) * len(i_list) * len(j_list) * len(k_list) * len(n_iter_no_change_list) * \
    #         len(epsilon_list) * len(learning_rate_init_list) * len(solver_list) * len(activation_list)

    with Manager() as manager:
        q_metrics = manager.Queue()
        jobs = []

        with Pool(no_threads) as pool:
            watcher = pool.apply_async(listener_write_to_file, (q_metrics, my_filename))
            for max_iter in max_iter_list:
                for i in i_list:
                    for j in j_list:
                        for k in k_list:
                            for solver in solver_list:
                                for activation in activation_list:
                                    job = pool.apply_async(run_algorithm_MLP_configuration_parallel, (X, y, q_metrics,
                                                                                                      max_iter,
                                                                                                      [i, j, k],
                                                                                                      123, 1,
                                                                                                      activation,
                                                                                                      solver, 0.0001,
                                                                                                      'constant', 0.001,
                                                                                                      0.5, True, 0.9,
                                                                                                      True, False,
                                                                                                      0.1, 0.9, 0.999,
                                                                                                      1e-8, 10,
                                                                                                      15000, train_size,
                                                                                                      stratify))
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


def create_label_for_MLP_for_row(row):
    return create_label_for_MLP(row['max_iter'], row['hidden_layer_sizes'], row['random_state'], row['batch_size'],
                                row['activation'], row['solver'], row['alpha'], row['learning_rate'],
                                row['learning_rate_init'], row['power_t'], row['shuffle'], row['momentum'],
                                row['nesterovs_momentum'], row['early_stopping'], row['validation_fraction'],
                                row['beta_1'], row['beta_2'], row['epsilon'], row['n_iter_no_change'], row['max_fun'])


def create_label_for_MLP(max_iter, hidden_layer_sizes, random_state, batch_size, activation, solver, alpha,
                         learning_rate, learning_rate_init, power_t, shuffle, momentum, nesterovs_momentum,
                         early_stopping, validation_fraction, beta_1, beta_2, epsilon, n_iter_no_change,
                         max_fun):
    return "MLP, max_iter = " + str(max_iter) + ", hidden layer sizes=" + str(hidden_layer_sizes) + ", random_state=" + \
           str(random_state) + ", batch_size=" + str(batch_size) + ", activation=" + str(
        activation) + ", solver=" + str(solver) + \
           ", alpha=" + str(alpha) + ", learning_rate=" + str(learning_rate) + ", learning_rate_init=" + str(
        learning_rate_init) + \
           ", power_t=" + str(power_t) + ", shuffle=" + str(shuffle) + ", momentum=" + str(
        momentum) + ", nesterovs_momentum" + \
           str(nesterovs_momentum) + ", early_stopping=" + str(early_stopping) + ", validation_fraction=" + str(
        validation_fraction) + \
           ", beta_1=" + str(beta_1) + ", beta_2=" + str(beta_2) + ", epsilon=" + str(
        epsilon) + ", n_iter_no_change=" + str(n_iter_no_change) + \
           ", max_fun=" + str(max_fun)


def run_algorithm_MLP_configuration_parallel(X, y, q_metrics,
                                             max_iter=100, hidden_layer_sizes=[35, 100, 100],
                                             random_state=123, batch_size=1, activation='relu',
                                             solver='adam', alpha=0.0001, learning_rate='constant',
                                             learning_rate_init=0.001, power_t=0.5, shuffle=True,
                                             momentum=0.9, nesterovs_momentum=True, early_stopping=False,
                                             validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                                             epsilon=1e-8, n_iter_no_change=10, max_fun=15000,
                                             train_size=0.8, stratify=False):
    X_train, X_test, y_train, y_test = split_data_in_testing_training(X, y, stratify, train_size)

    try:
        # Creating the classifier object
        classifier = MLPClassifier(max_iter=max_iter, hidden_layer_sizes=hidden_layer_sizes, random_state=random_state,
                                   batch_size=batch_size,
                                   activation=activation, solver=solver, alpha=alpha, learning_rate=learning_rate,
                                   learning_rate_init=learning_rate_init, power_t=power_t, shuffle=shuffle,
                                   momentum=momentum, nesterovs_momentum=nesterovs_momentum,
                                   early_stopping=early_stopping,
                                   validation_fraction=validation_fraction, beta_1=beta_1, beta_2=beta_2,
                                   epsilon=epsilon, n_iter_no_change=n_iter_no_change, max_fun=max_fun)

        # Performing training
        classifier.fit(X_train, y_train)

        # Make predictions
        y_pred, y_pred_probabilities = prediction(X_test, classifier)

        # Compute metrics
        label = create_label_for_MLP(max_iter, hidden_layer_sizes, random_state, batch_size, activation, solver, alpha,
                                     learning_rate, learning_rate_init, power_t, shuffle, momentum, nesterovs_momentum,
                                     early_stopping, validation_fraction, beta_1, beta_2, epsilon, n_iter_no_change,
                                     max_fun)
        precision, recall, f1, roc_auc = cal_metrics(y_test, y_pred, y_pred_probabilities, label, classifier)
        string_results_for_queue = convert_metrics_to_csv(',', label, max_iter, hidden_layer_sizes, random_state,
                                                          batch_size, activation, solver, alpha, learning_rate,
                                                          learning_rate_init,
                                                          power_t, shuffle, momentum, nesterovs_momentum,
                                                          early_stopping,
                                                          validation_fraction, beta_1, beta_2, epsilon,
                                                          n_iter_no_change,
                                                          max_fun, precision, recall, f1, roc_auc)
        q_metrics.put(string_results_for_queue)
    except Exception as err:
        print(err)
    except RuntimeWarning as warn:
        print(warn)


def run_algorithm_MLP(filename='', path='', stratify=False, train_size=0.8,
                      normalize_data=False, scaler='min-max'):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_MLP()

    # full_path_filename = '/content/drive/MyDrive/code/' + filename
    # path_to_script = os.path.dirname(os.path.abspath(__file__))
    my_filename = os.path.join(path, 'results/mlp', filename)

    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_MLP, header=True)

    # GINI - Equivalent for islam2019mapreduce
    # run_algorithm_MLP_configuration(metrics,
    #                                 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1',
    #                                 X, y, stratify=stratify, train_size=train_size)
    #
    # # CALIBRATING max_iter param
    # for max_iter in range(50, 150, 25):
    #     run_algorithm_MLP_configuration(metrics, 'MLP, max_iter = ' + str(
    #         max_iter) + ', hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1', X, y,
    #                                     max_iter=max_iter, stratify=stratify, train_size=train_size)
    #
    # # CALIBRATING batch_size param
    # for batch_size in range(10, 100, 10):
    #     run_algorithm_MLP_configuration(metrics,
    #                                     'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = ' + str(
    #                                         batch_size), X, y, batch_size=batch_size, stratify=stratify,
    #                                     train_size=train_size)
    #
    # # CALIBRATING activation param
    # run_algorithm_MLP_configuration(metrics,
    #                                 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, activation = tanh',
    #                                 X, y, activation='tanh', stratify=stratify, train_size=train_size)
    # run_algorithm_MLP_configuration(metrics,
    #                                 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, activation = logistic',
    #                                 X, y, activation='logistic', stratify=stratify, train_size=train_size)
    # run_algorithm_MLP_configuration(metrics,
    #                                 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, activation = identity',
    #                                 X, y, activation='identity', stratify=stratify, train_size=train_size)
    # metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_MLP)
    #
    # # CALIBRATING solver param
    # run_algorithm_MLP_configuration(metrics,
    #                                 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, solver = sgd',
    #                                 X, y, solver='sgd', stratify=stratify, train_size=train_size)
    # run_algorithm_MLP_configuration(metrics,
    #                                 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, solver = lbfgs',
    #                                 X, y, solver='lbfgs', stratify=stratify, train_size=train_size)
    #
    # # CALIBRATING alpha param
    # for alpha in np.arange(0.00001, 0.0002, 0.00001):
    #     run_algorithm_MLP_configuration(metrics,
    #                                     'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, alpha = ' + str(
    #                                         alpha), X, y, alpha=alpha, stratify=stratify, train_size=train_size)
    #
    # # CALIBRATING learning_rate param
    # run_algorithm_MLP_configuration(metrics,
    #                                 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, solver = sgd, learning_rate = constant',
    #                                 X, y, solver='sgd', learning_rate='constant', stratify=stratify,
    #                                 train_size=train_size)
    # run_algorithm_MLP_configuration(metrics,
    #                                 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, solver = sgd, learning_rate = invscaling',
    #                                 X, y, solver='sgd', learning_rate='invscaling', stratify=stratify,
    #                                 train_size=train_size)
    # run_algorithm_MLP_configuration(metrics,
    #                                 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, solver = sgd, learning_rate = adaptive',
    #                                 X, y, solver='sgd', learning_rate='adaptive', stratify=stratify,
    #                                 train_size=train_size)
    #
    # # CALIBRATING learning_rate_init param
    # for learning_rate_init in [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.0011,
    #                            0.0012, 0.0013, 0.0014, 0.0015, 0.0016, 0.0017, 0.0018, 0.0019]:
    #     run_algorithm_MLP_configuration(metrics,
    #                                     'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, solver = sgd, learning_rate_init = ' + str(
    #                                         learning_rate_init), X, y, solver='sgd',
    #                                     learning_rate_init=learning_rate_init, stratify=stratify, train_size=train_size)
    #     run_algorithm_MLP_configuration(metrics,
    #                                     'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, solver = adam, learning_rate_init = ' + str(
    #                                         learning_rate_init), X, y, solver='adam',
    #                                     learning_rate_init=learning_rate_init, stratify=stratify, train_size=train_size)
    # metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_MLP)
    # # CALIBRATING power_t param
    # # for power_t in np.arange(0.4, 0.6, 0.05):
    # for power_t in [0.4, 0.45, 0.55, 0.6]:
    #     run_algorithm_MLP_configuration(metrics,
    #                                     'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, solver = sgd, power_t = ' + str(
    #                                         power_t), X, y, solver='sgd', power_t=power_t, stratify=stratify,
    #                                     train_size=train_size)
    #
    # # CALIBRATING shuffle param
    # run_algorithm_MLP_configuration(metrics,
    #                                 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, shuffle = True',
    #                                 X, y, shuffle=True, stratify=stratify, train_size=train_size)
    # run_algorithm_MLP_configuration(metrics,
    #                                 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, shuffle = False',
    #                                 X, y, shuffle=False, stratify=stratify, train_size=train_size)
    #
    # # CALIBRATING momentum param
    # for momentum in np.arange(0.75, 1.0, 0.05):
    #     run_algorithm_MLP_configuration(metrics,
    #                                     'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, solver = sgd, momentum = ' + str(
    #                                         momentum) + ', nesterovs_momentum = True', X, y, solver='sgd',
    #                                     momentum=momentum, nesterovs_momentum=True, stratify=stratify,
    #                                     train_size=train_size)
    #     run_algorithm_MLP_configuration(metrics,
    #                                     'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, solver = sgd, momentum = ' + str(
    #                                         momentum) + ', nesterovs_momentum = False', X, y, solver='sgd',
    #                                     momentum=momentum, nesterovs_momentum=False, stratify=stratify,
    #                                     train_size=train_size)
    #
    # # CALIBRATING early_stopping param
    # run_algorithm_MLP_configuration(metrics,
    #                                 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, early_stopping = True, solver = sgd',
    #                                 X, y, solver='sgd', early_stopping=True, stratify=stratify, train_size=train_size)
    # run_algorithm_MLP_configuration(metrics,
    #                                 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, early_stopping = False, solver = sgd',
    #                                 X, y, solver='sgd', early_stopping=False, stratify=stratify, train_size=train_size)
    # run_algorithm_MLP_configuration(metrics,
    #                                 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, early_stopping = True, solver = adam',
    #                                 X, y, solver='adam', early_stopping=True, stratify=stratify, train_size=train_size)
    # run_algorithm_MLP_configuration(metrics,
    #                                 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, early_stopping = False, solver = adam',
    #                                 X, y, solver='adam', early_stopping=False, stratify=stratify, train_size=train_size)
    # metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_MLP)
    #
    # # CALIBRATING validation_fraction param
    # for validation_fraction in np.arange(0.05, 0.25, 0.05):
    #     run_algorithm_MLP_configuration(metrics,
    #                                     'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, early_stopping = True, solver = sgd, validation_fraction = ' + str(
    #                                         validation_fraction), X, y, solver='sgd', early_stopping=True,
    #                                     validation_fraction=validation_fraction, stratify=stratify,
    #                                     train_size=train_size)
    #     run_algorithm_MLP_configuration(metrics,
    #                                     'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, early_stopping = True, solver = adam, validation_fraction = ' + str(
    #                                         validation_fraction), X, y, solver='adam', early_stopping=True,
    #                                     validation_fraction=validation_fraction, stratify=stratify,
    #                                     train_size=train_size)
    #
    # # CALIBRATING beta_1 param
    # for beta_1 in np.arange(0.75, 1.0, 0.05):
    #     run_algorithm_MLP_configuration(metrics,
    #                                     'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, solver = adam, beta_1 = ' + str(
    #                                         beta_1), X, y, solver='adam', beta_1=beta_1, stratify=stratify,
    #                                     train_size=train_size)
    #
    # # CALIBRATING beta_2 param
    # for beta_2 in np.arange(0.75, 1.0, 0.05):
    #     run_algorithm_MLP_configuration(metrics,
    #                                     'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, solver = adam, beta_2 = ' + str(
    #                                         beta_2), X, y, solver='adam', beta_2=beta_2, stratify=stratify,
    #                                     train_size=train_size)
    #
    # # CALIBRATING epsilon param
    # for epsilon in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15,
    #                 1e-16, 1e-17]:
    #     run_algorithm_MLP_configuration(metrics,
    #                                     'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, solver = adam, epsilon = ' + str(
    #                                         epsilon), X, y, solver='adam', epsilon=epsilon, stratify=stratify,
    #                                     train_size=train_size)
    #
    # # CALIBRATING n_iter_no_change param
    # for n_iter_no_change in chain(range(1, 20, 1), range(20, 50, 5), range(50, 100, 10), range(100, 500, 20),
    #                                range(500, 1000, 100)):
    #     run_algorithm_MLP_configuration(metrics,
    #                                     'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, solver = adam, n_iter_no_change = ' + str(
    #                                         n_iter_no_change), X, y, solver='adam',
    #                                     n_iter_no_change=n_iter_no_change, stratify=stratify,
    #                                     train_size=train_size)
    #     run_algorithm_MLP_configuration(metrics,
    #                                     'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, solver = sgd, n_iter_no_change = ' + str(
    #                                         n_iter_no_change), X, y, solver='sgd', n_iter_no_change=n_iter_no_change,
    #                                     stratify=stratify,
    #                                     train_size=train_size)
    #
    # # CALIBRATING max_fun param
    # for max_fun in chain(range(10000, 14000, 500), range(14000, 16000, 100), range(16000, 20000, 500)):
    #     run_algorithm_MLP_configuration(metrics,
    #                                     'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, solver = lbfgs, max_fun = ' + str(
    #                                         max_fun), X, y, solver='lbfgs', max_fun=max_fun, stratify=stratify,
    #                                     train_size=train_size)
    #
    # metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_MLP)
    #
    # # CALIBRATING hidden_layer_sizes param
    # for i in range(25, 200, 25):
    #     for j in range(25, 200, 25):
    #         for k in range(25, 200, 25):
    #             run_algorithm_MLP_configuration(metrics,
    #                                             'MLP, max_iter = 100, hidden_layer_sizes = [' + str(i) + ', ' + str(
    #                                                 j) + ', ' + str(k) + '], random_state = 123, batch_size = 1', X, y,
    #                                             hidden_layer_sizes=[i, j, k], stratify=stratify, train_size=train_size)
    #         metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_MLP)

    for max_iter in range(50, 175, 25):
        for i in range(10, 200, 5):
            for j in range(10, 200, 5):
                for k in range(10, 200, 5):
                    for n_iter_no_change in range(5, 20, 1):
                        for epsilon in [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-14, 1e-15, 1e-16, 1e-17, 1e-18,
                                        1e-19, 1e-20]:
                            for learning_rate_init in [0.0005, 0.00055, 0.0006, 0.00065, 0.0007, 0.00075, 0.0008,
                                                       0.00085,
                                                       0.0009, 0.00095, 0.001, 0.0011, 0.0012, 0.0013, 0.0014, 0.0015,
                                                       0.0016, 0.00165, 0.0017, 0.00175, 0.0018, 0.00185, 0.0019,
                                                       0.00195,
                                                       0.002]:
                                for solver in ['adam', 'sgd']:
                                    for activation in ['relu', 'identity', 'logistic', 'tanh']:
                                        run_algorithm_MLP_configuration(metrics,
                                                                        'MLP, activation=' + activation + ', solver = ' + solver + ', learning_rate_init=' + str(
                                                                            learning_rate_init) + ', epsilon=' + str(
                                                                            epsilon) + ', n_iter_no_change=' + str(
                                                                            n_iter_no_change) + ', max_iter=' + str(
                                                                            max_iter) + ', hidden_layer_sizes = ' + str(
                                                                            [i, j, k]), X, y,
                                                                        max_fun=15000, validation_fraction=0.1,
                                                                        beta_1=0.9, beta_2=0.999, early_stopping=False,
                                                                        nesterovs_momentum=True, momentum=0.9,
                                                                        shuffle=True, power_t=0.5,
                                                                        learning_rate='constant',
                                                                        alpha=0.001,
                                                                        batch_size=1,
                                                                        activation=activation, solver=solver,
                                                                        learning_rate_init=learning_rate_init,
                                                                        epsilon=epsilon,
                                                                        n_iter_no_change=n_iter_no_change,
                                                                        max_iter=max_iter, hidden_layer_sizes=[i, j, k],
                                                                        stratify=stratify, train_size=train_size)

                                metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_MLP)

    for max_iter in range(50, 175, 25):
        for i in range(10, 200, 5):
            for j in range(10, 200, 5):
                for n_iter_no_change in range(5, 20, 1):
                    for epsilon in [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-14, 1e-15, 1e-16, 1e-17, 1e-18,
                                    1e-19, 1e-20]:
                        for learning_rate_init in [0.0005, 0.00055, 0.0006, 0.00065, 0.0007, 0.00075, 0.0008,
                                                   0.00085,
                                                   0.0009, 0.00095, 0.001, 0.0011, 0.0012, 0.0013, 0.0014, 0.0015,
                                                   0.0016, 0.00165, 0.0017, 0.00175, 0.0018, 0.00185, 0.0019,
                                                   0.00195,
                                                   0.002]:
                            for solver in ['adam', 'sgd']:
                                for activation in ['relu', 'identity', 'logistic', 'tanh']:
                                    run_algorithm_MLP_configuration(metrics,
                                                                    'MLP, activation=' + activation + ', solver = ' + solver + ', learning_rate_init=' + str(
                                                                        learning_rate_init) + ', epsilon=' + str(
                                                                        epsilon) + ', n_iter_no_change=' + str(
                                                                        n_iter_no_change) + ', max_iter=' + str(
                                                                        max_iter) + ', hidden_layer_sizes = ' + str(
                                                                        [i, j]), X, y,
                                                                    max_fun=15000, validation_fraction=0.1,
                                                                    beta_1=0.9, beta_2=0.999, early_stopping=False,
                                                                    nesterovs_momentum=True, momentum=0.9,
                                                                    shuffle=True, power_t=0.5,
                                                                    learning_rate='constant',
                                                                    alpha=0.001,
                                                                    batch_size=1,
                                                                    activation=activation, solver=solver,
                                                                    learning_rate_init=learning_rate_init,
                                                                    epsilon=epsilon,
                                                                    n_iter_no_change=n_iter_no_change,
                                                                    max_iter=max_iter, hidden_layer_sizes=[i, j],
                                                                    stratify=stratify, train_size=train_size)

                            metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_MLP)

    metrics = appendMetricsTOCSV(my_filename, metrics, init_metrics_for_MLP)


# # export metrics to CSV FILE
# df_metrics = pd.DataFrame(metrics)
# df_metrics.to_csv(filename, encoding='utf-8', index= True)


# label = '\"MLP, max_iter=' + str(max_iter) + ', hidden_layer_sizes=' + str(
#     hidden_layer_sizes) + ', batch_size = ' + str(
#     batch_size) + ', activation = ' + activation + ', solver=' + solver + ", alpha=" + str(
#     alpha) + ', learning_rate=' + str(
#     learning_rate) + ', learning_rate_init=' + str(learning_rate_init) + ', power_t=' + str(
#     power_t) + ', shuffle=' + str(shuffle) + ', momentum=' + str(momentum) + ', nesterovs_momentum=' + str(
#     nesterovs_momentum) + ', early_stopping=' + str(early_stopping) + ', validation_fraction=' + str(
#     validation_fraction) + ', beta_1=' + str(beta_1) + 'beta_2=' + str(beta_2) + ', epsilon=' + str(
# epsilon) + ', n_iter_no_change=' + str(n_iter_no_change) + 'max_fun=' + str(max_fun)+'\"'


def prepare_MLP_params(row):
    params_dict = {}
    #  hidden_layer_sizes=[35, 100, 100]
    fields = row['hidden_layer_sizes'].replace('[', '').replace(']', '').split(',')
    params_dict['hidden_layer_sizes'] = [int(i) for i in fields]

    if row['batch_size'] == 'auto':
        params_dict['batch_size'] = 'auto'
    else:
        params_dict['batch_size'] = int(row['batch_size'])

    if row['shuffle'] == 'TRUE':
        params_dict['shuffle'] = True
    else:
        params_dict['shuffle'] = False

    if row['nesterovs_momentum'] == 'TRUE':
        params_dict['nesterovs_momentum'] = True
    else:
        params_dict['nesterovs_momentum'] = False

    if row['early_stopping'] == 'TRUE':
        params_dict['early_stopping'] = True
    else:
        params_dict['early_stopping'] = False

    params_dict['max_iter'] = int(row['max_iter'])
    params_dict['random_state'] = int(row['random_state'])
    params_dict['alpha'] = float(row['alpha'])
    params_dict['activation'] = row['activation']
    params_dict['solver'] = row['solver']
    params_dict['learning_rate'] = row['learning_rate']
    params_dict['learning_rate_init'] = float(row['learning_rate_init'])
    params_dict['power_t'] = float(row['power_t'])
    params_dict['momentum'] = float(row['momentum'])
    params_dict['validation_fraction'] = float(row['validation_fraction'])
    params_dict['beta_1'] = float(row['beta_1'])
    params_dict['beta_2'] = float(row['beta_2'])
    params_dict['epsilon'] = float(row['epsilon'])
    params_dict['n_iter_no_change'] = int(row['n_iter_no_change'])
    params_dict['max_fun'] = int(row['max_fun'])

    return params_dict


def run_best_configs_mlp(df_configs, filename='', path='', stratify=True, train_size=0.8,
                         normalize_data=True, scaler='min-max', n_rep=100):
    y, X = load_normalized_dataset(file=None, normalize=normalize_data, scaler=scaler)
    metrics = init_metrics_for_MLP()
    my_filename = os.path.join(path, 'new_results/mlp', filename)

    for i in range(1, n_rep):
        for index, row in df_configs.iterrows():
            # print('index' + str(index))
            # print(row)
            label = create_label_for_MLP_for_row(row)
            params = prepare_MLP_params(row)
            run_algorithm_MLP_configuration(metrics, label, X, y,
                                            max_iter=params['max_iter'],
                                            hidden_layer_sizes=params['hidden_layer_sizes'],
                                            random_state=params['random_state'], batch_size=params['batch_size'],
                                            activation=params['activation'], solver=params['solver'],
                                            alpha=params['alpha'], learning_rate=params['learning_rate'],
                                            learning_rate_init=params['learning_rate_init'], power_t=params['power_t'],
                                            shuffle=params['shuffle'], momentum=params['momentum'],
                                            nesterovs_momentum=params['nesterovs_momentum'],
                                            early_stopping=params['early_stopping'],
                                            validation_fraction=params['validation_fraction'], beta_1=params['beta_1'],
                                            beta_2=params['beta_2'], epsilon=params['epsilon'],
                                            n_iter_no_change=params['n_iter_no_change'], max_fun=params['max_fun'],
                                            stratify=stratify, train_size=train_size
                                            )

    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.groupby(['label'], as_index=False).agg({'precision': 'mean', 'recall': 'mean',
                                                                    'f1_score': 'mean', 'roc_auc': 'mean',
                                                                    'max_iter': 'first', 'hidden_layer_sizes': 'first',
                                                                    'random_state': 'first', 'batch_size': 'first',
                                                                    'activation': 'first', 'solver': 'first',
                                                                    'alpha': 'first', 'learning_rate': 'first',
                                                                    'learning_rate_init': 'first', 'power_t': 'first',
                                                                    'shuffle': 'first', 'momentum': 'first',
                                                                    'nesterovs_momentum': 'first',
                                                                    'early_stopping': 'first',
                                                                    'validation_fraction': 'first',
                                                                    'beta_1': 'first', 'beta_2': 'first',
                                                                    'epsilon': 'first', 'n_iter_no_change': 'first',
                                                                    'max_fun': 'first'
                                                                    })
    metrics_df = compute_average_metric(metrics_df)
    metrics_df.sort_values(by=['average_metric'], ascending=False, inplace=True)
    metrics = appendMetricsTOCSV(my_filename, metrics_df, init_metrics_for_MLP, header=True)


def create_MLP_classifier(row):
    params = prepare_MLP_params(row)
    classifier = MLPClassifier(max_iter=params['max_iter'], hidden_layer_sizes=params['hidden_layer_sizes'],
                               random_state=params['random_state'], batch_size=params['batch_size'],
                               activation=params['activation'], solver=params['solver'], alpha=params['alpha'],
                               learning_rate=params['learning_rate'], learning_rate_init=params['learning_rate_init'],
                               power_t=params['power_t'], shuffle=params['shuffle'], momentum=params['momentum'],
                               nesterovs_momentum=params['nesterovs_momentum'], early_stopping=params['early_stopping'],
                               validation_fraction=params['validation_fraction'], beta_1=params['beta_1'],
                               beta_2=params['beta_2'], epsilon=params['epsilon'],
                               n_iter_no_change=params['n_iter_no_change'], max_fun=params['max_fun'])
    return classifier


