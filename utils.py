import os
from datetime import datetime
from time import sleep

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler



def split_data(X, y, normalize_data = False, stratify = False, train_size = 0.8, scaler = 'min-max'):
    # # Split df into X and y
    # y = df['label'].copy()
    # X = df.drop('label', axis=1).copy()

    # Normalize data along the columns
    # d = preprocessing.normalize(X, axis = 0)
    # scaled_df = pd.DataFrame(d, columns=names)
    # scaled_df.head()

    if (normalize_data == True):
        if scaler == 'min-max':
            # Normalize with MinMaxScaler
            scaler = MinMaxScaler()
        elif scaler == 'standard':
            # Normalize with StandardScaler
            scaler = StandardScaler()
        names = X.columns
        d = scaler.fit_transform(X)
        scaled_df = pd.DataFrame(d, columns=names)
        # print(scaled_df.head())
    else:
        scaled_df = X.copy(deep = True)

    # Train-test split
    if (stratify == True):
        X_train, X_test, y_train, y_test = train_test_split(scaled_df, y, stratify=y, train_size=train_size)
    else:
        X_train, X_test, y_train, y_test = train_test_split(scaled_df, y, train_size=train_size)

    return X_train, X_test, y_train, y_test



def data_normalization(X, y, stratify = False, train_size = 0.8):
    # Split df into X and y
    # y = df['label'].copy()
    # X = df.drop('label', axis=1).copy()

    # Normalize data along the columns
    # d = preprocessing.normalize(X, axis = 0)
    # scaled_df = pd.DataFrame(d, columns=names)
    # scaled_df.head()


    # Normalize with MinMaxScaler
    scaler = preprocessing.MinMaxScaler()
    names = X.columns
    d = scaler.fit_transform(X)
    scaled_df = pd.DataFrame(d, columns=names)

    # Train-test split
    if (stratify):
        X_train, X_test, y_train, y_test = train_test_split(scaled_df, y, stratify=y, train_size = train_size)
    else:
        X_train, X_test, y_train, y_test = train_test_split(scaled_df, y, train_size = train_size)


    return X_train, X_test, y_train, y_test




# Function to make predictions
def prediction(X_test, clf_object):

    # Predicton on test
    y_pred = clf_object.predict(X_test)
    y_pred_probabilities = clf_object.predict_proba(X_test)
    # print("Predicted values:")
    # print(y_pred)
    # print("Predicted probabilities: ")
    # print(y_pred_probabilities)
    return y_pred, y_pred_probabilities

# Function to calculate accuracy
def cal_metrics(y_test, y_pred, y_pred_probabilities, classifier):

    # print("\n-----------------------------------------------")
    # print("METRICS FOR "+ classifier)
    # print("-----------------------------------------------\n")

    # print("Confusion Matrix: ",
    #     confusion_matrix(y_test, y_pred))

    precision = precision_score(y_test,y_pred)*100
    # print ("Precision : ", precision)

    recall = recall_score(y_test,y_pred)*100
    # print ("Recall : ", recall)

    f1 = f1_score(y_test,y_pred,  average='weighted', labels=np.unique(y_pred))*100
    # print ("F1 score : ",f1)

    roc_auc = 0.0
    if (len(y_pred_probabilities) > 0):
        try:
            roc_auc = roc_auc_score(y_test, y_pred_probabilities[:,1], average='weighted', labels=np.unique(y_pred))
            # print ("ROC_AUC score: ", roc_auc)
        except:
            pass

    # fpr - false positive rate, tpr - true positive rate, thresholds values based on which the class 0 or 1 is chosen
    # fpr, tpr, thresholds = roc_curve(y_test, y_pred_probabilities[:,1])
    # plt.plot(fpr, tpr, marker = '.', label = "AUROC = %0.3f" %roc_auc)
    # plt.title("ROC curve")
    # plt.xlabel('False Positives Rate')
    # plt.ylabel('True POsitives Rate')
    # plt.legend()
    # plt.show()

    # print("Report : ",
    # classification_report(y_test, y_pred))

    return precision, recall, f1, roc_auc


def appendMetricsTOCSV(filename, metrics, init_function, header = False, ):
    df = pd.DataFrame(metrics)
    # append data frame to CSV file
    try:
        df.to_csv(filename, mode='a', encoding='utf-8', index=False, header=header)
    except:
        sleep(60)
        df.to_csv(filename, mode='a', encoding='utf-8', index=False, header=header)

    return init_function()



def run_algorithm_mlp_configuration(metrics, label, X, y,
                                    max_iter = 100, hidden_layer_sizes = [35, 100, 100],
                                    random_state = 123, batch_size = 1, activation = 'relu',
                                    solver = 'adam', alpha = 0.0001, learning_rate = 'constant',
                                    learning_rate_init = 0.001, power_t = 0.5, shuffle = True,
                                    momentum = 0.9, nesterovs_momentum = True, early_stopping = False,
                                    validation_fraction = 0.1, beta_1 = 0.9, beta_2 = 0.999,
                                    train_size = 0.8, stratify = False, normalize_data = False,
                                    log_file = 'logs.txt'):
    if (normalize_data == True):
        X_train, X_test, y_train, y_test = data_normalization(X, y, stratify = stratify, train_size = train_size)
    else:
        if (stratify == True):
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, train_size=train_size)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)

    # Creating the classifier object
    classifier = MLPClassifier(max_iter=max_iter, hidden_layer_sizes=hidden_layer_sizes, random_state=random_state, batch_size = batch_size,
                               activation = activation, solver = solver, alpha = alpha, learning_rate = learning_rate,
                               learning_rate_init= learning_rate_init, power_t = power_t, shuffle = shuffle,
                               momentum=momentum, nesterovs_momentum=nesterovs_momentum, early_stopping=early_stopping,
                               validation_fraction=validation_fraction, beta_1=beta_1, beta_2=beta_2)

    try:
        # Performing training
        classifier.fit(X_train, y_train)
        # Make predictions
        y_pred, y_pred_probabilities = prediction(X_test, classifier)

        # Compute metrics
        precision, recall, f1, roc_auc = cal_metrics(y_test, y_pred, y_pred_probabilities, label)
        metrics['label'].append(label)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1_score'].append(f1)
        metrics['roc_auc'].append(roc_auc)
    except Exception as e:
        write_error_to_log_file(e, log_file)




def run_algorithm_mlp(df, filename = '', stratify = False, train_size = 0.8, normalize_data = False):
    y = df['label'].copy()
    X = df.drop('label', axis=1).copy()
    metrics = {'label': [],
               'precision': [],
               'recall': [],
               'f1_score': [],
               'roc_auc': []
               }

    # GINI - Equivalent for islam2019mapreduce
    run_algorithm_mlp_configuration(metrics, 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1', X, y, stratify = stratify, train_size = train_size, normalize_data = normalize_data)

    # CALIBRATING max_iter param
    for max_iter in range(50, 150, 25):
        run_algorithm_mlp_configuration(metrics, 'MLP, max_iter = '+str(max_iter)+', hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1', X, y, max_iter = max_iter, stratify = stratify, train_size = train_size, normalize_data = normalize_data)

    # CALIBRATING batch_size param
    for batch_size in range(10, 100, 10):
        run_algorithm_mlp_configuration(metrics, 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = '+str(batch_size), X, y, batch_size = batch_size, stratify = stratify, train_size = train_size, normalize_data = normalize_data)

    # CALIBRATING activation param
    run_algorithm_mlp_configuration(metrics, 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, activation = tanh', X, y, activation = 'tanh', stratify = stratify, train_size = train_size, normalize_data = normalize_data)
    run_algorithm_mlp_configuration(metrics, 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, activation = logistic', X, y, activation = 'logistic', stratify = stratify, train_size = train_size, normalize_data = normalize_data)
    run_algorithm_mlp_configuration(metrics, 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, activation = identity', X, y, activation = 'identity', stratify = stratify, train_size = train_size, normalize_data = normalize_data)


    # CALIBRATING solver param
    run_algorithm_mlp_configuration(metrics, 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, solver = sgd', X, y, solver = 'sgd', stratify = stratify, train_size = train_size, normalize_data = normalize_data)
    run_algorithm_mlp_configuration(metrics, 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, solver = lbfgs', X, y, solver = 'lbfgs', stratify = stratify, train_size = train_size, normalize_data = normalize_data)

    # CALIBRATING alpha param
    for alpha in np.arange(0.00001, 0.0002, 0.00001):
        run_algorithm_mlp_configuration(metrics, 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, alpha = '+str(alpha), X, y, alpha = alpha, stratify = stratify, train_size = train_size, normalize_data = normalize_data)

    # CALIBRATING learning_rate param
    run_algorithm_mlp_configuration(metrics, 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, solver = sgd, learning_rate = constant', X, y, solver = 'sgd', learning_rate='constant', stratify = stratify, train_size = train_size, normalize_data = normalize_data)
    run_algorithm_mlp_configuration(metrics, 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, solver = sgd, learning_rate = invscaling', X, y, solver = 'sgd', learning_rate='invscaling', stratify = stratify, train_size = train_size, normalize_data = normalize_data)
    run_algorithm_mlp_configuration(metrics, 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, solver = sgd, learning_rate = adaptive', X, y, solver = 'sgd', learning_rate='adaptive', stratify = stratify, train_size = train_size, normalize_data = normalize_data)


    # CALIBRATING learning_rate_init param
    for learning_rate_init in [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.0011, 0.0012, 0.0013, 0.0014, 0.0015, 0.0016, 0.0017, 0.0018, 0.0019]:
        run_algorithm_mlp_configuration(metrics, 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, solver = sgd, learning_rate_init = '+str(learning_rate_init), X, y, solver = 'sgd', learning_rate_init=learning_rate_init, stratify = stratify, train_size = train_size, normalize_data = normalize_data)
        run_algorithm_mlp_configuration(metrics, 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, solver = adam, learning_rate_init = '+str(learning_rate_init), X, y, solver = 'adam', learning_rate_init=learning_rate_init, stratify = stratify, train_size = train_size, normalize_data = normalize_data)


    # CALIBRATING power_t param
    # for power_t in np.arange(0.4, 0.6, 0.05):
    for power_t in [0.4, 0.45, 0.55, 0.6]:
        run_algorithm_mlp_configuration(metrics, 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, solver = sgd, power_t = '+str(power_t), X, y, solver = 'sgd', power_t=power_t, stratify = stratify, train_size = train_size, normalize_data = normalize_data)

    # CALIBRATING shuffle param
    run_algorithm_mlp_configuration(metrics, 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, shuffle = True', X, y, shuffle=True, stratify = stratify, train_size = train_size, normalize_data = normalize_data)
    run_algorithm_mlp_configuration(metrics, 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, shuffle = False', X, y, shuffle=False, stratify = stratify, train_size = train_size, normalize_data = normalize_data)


    # CALIBRATING momentum param
    for momentum in np.arange(0.75, 1.0, 0.05):
        run_algorithm_mlp_configuration(metrics, 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, solver = sgd, momentum = '+str(momentum)+', nesterovs_momentum = True', X, y, solver = 'sgd', momentum=momentum, nesterovs_momentum = True, stratify = stratify, train_size = train_size, normalize_data = normalize_data)
        run_algorithm_mlp_configuration(metrics, 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, solver = sgd, momentum = '+str(momentum)+', nesterovs_momentum = False', X, y, solver = 'sgd', momentum=momentum, nesterovs_momentum = False, stratify = stratify, train_size = train_size, normalize_data = normalize_data)

    # CALIBRATING early_stopping param
    run_algorithm_mlp_configuration(metrics, 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, early_stopping = True, solver = sgd', X, y, solver = 'sgd', early_stopping=True, stratify = stratify, train_size = train_size, normalize_data = normalize_data)
    run_algorithm_mlp_configuration(metrics, 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, early_stopping = False, solver = sgd', X, y, solver = 'sgd', early_stopping=False, stratify = stratify, train_size = train_size, normalize_data = normalize_data)
    run_algorithm_mlp_configuration(metrics, 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, early_stopping = True, solver = adam', X, y, solver = 'adam', early_stopping=True, stratify = stratify, train_size = train_size, normalize_data = normalize_data)
    run_algorithm_mlp_configuration(metrics, 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, early_stopping = False, solver = adam', X, y, solver = 'adam', early_stopping=False, stratify = stratify, train_size = train_size, normalize_data = normalize_data)

    # CALIBRATING validation_fraction param
    for validation_fraction in np.arange(0.05, 0.25, 0.05):
        run_algorithm_mlp_configuration(metrics, 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, early_stopping = True, solver = sgd, validation_fraction = '+str(validation_fraction), X, y, solver = 'sgd', early_stopping=True, validation_fraction= validation_fraction, stratify = stratify, train_size = train_size, normalize_data = normalize_data)
        run_algorithm_mlp_configuration(metrics, 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, early_stopping = True, solver = adam, validation_fraction = ' + str(validation_fraction), X, y, solver = 'adam', early_stopping=True, validation_fraction = validation_fraction, stratify = stratify, train_size = train_size, normalize_data = normalize_data)

    # CALIBRATING beta_1 param
    for beta_1 in np.arange(0.75, 1.0, 0.05):
        run_algorithm_mlp_configuration(metrics, 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, solver = adam, beta_1 = ' + str(beta_1), X, y, solver = 'adam', beta_1 = beta_1, stratify = stratify, train_size = train_size, normalize_data = normalize_data)

    # CALIBRATING beta_2 param
    for beta_2 in np.arange(0.75, 1.0, 0.05):
        run_algorithm_mlp_configuration(metrics, 'MLP, max_iter = 100, hidden_layer_sizes = [35, 100, 100], random_state = 123, batch_size = 1, solver = adam, beta_2 = ' + str(beta_2), X, y, solver = 'adam', beta_2 = beta_2, stratify = stratify, train_size = train_size, normalize_data = normalize_data)

    #TODO: our proposed MLP

    # CALIBRATING hidden_layer_sizes param
    # for i in range(25, 200, 25):
    #     for j in range(25, 200, 25):
    #       for k in range(25, 200, 25):
    #           run_algorithm_mlp_configuration(metrics, 'MLP, max_iter = 100, hidden_layer_sizes = ['+str(i)+', '+str(j)+', '+str(k)+'], random_state = 123, batch_size = 1', X, y, hidden_layer_sizes=[i, j, k], stratify = stratify, train_size = train_size, normalize_data = normalize_data)


   # export metrics to CSV FILE
    df_metrics = pd.DataFrame(metrics)
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    my_filename = os.path.join(path_to_script, filename)
    df_metrics.to_csv(my_filename, encoding='utf-8', index= True)




def run_algorithm_SVC_poly_kernel_configuration(metrics, label, X, y,
                                                tol = 1e-4,
                                                C=1.0,
                                                shrinking = True,
                                                cache_size = 200,
                                                class_weight = None,
                                                max_iter = 1000,
                                                gamma = 'scale',
                                                degree = 3,
                                                coef0 = 0.0,
                                                stratify = False, train_size = 0.8,
                                                normalize_data = False, log_file = 'logs.txt'
                                                ):

    if (normalize_data):
        X_train, X_test, y_train, y_test = data_normalization(X, y, stratify = stratify, train_size = train_size)
    else:
        if (stratify == True):
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, train_size=train_size)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)


    # Creating the classifier object
    classifier = SVC( probability=True, kernel = 'poly', tol = tol,  shrinking=shrinking, cache_size = cache_size,
                      C = C, class_weight = class_weight, max_iter = max_iter,
                      degree = degree, gamma= gamma, coef0 = coef0)

    try:
        # Performing training
        classifier.fit(X_train, y_train)

        # Make predictions
        y_pred, y_pred_probabilities = prediction(X_test, classifier)

        # Compute metrics
        precision, recall, f1, roc_auc = cal_metrics(y_test, y_pred, y_pred_probabilities, label)
        metrics['label'].append(label)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1_score'].append(f1)
        metrics['roc_auc'].append(roc_auc)
    except Exception as e:
        write_error_to_log_file(e, log_file)



def run_algorithm_SVC_poly_kernel(df,filename='', stratify = False, train_size = 0.8, normalize_data = False):
    y = df['label'].copy()
    X = df.drop('label', axis=1).copy()
    metrics = {'label': [],
               'precision': [],
               'recall': [],
               'f1_score': [],
               'roc_auc': []
               }

    # default algorithm
    run_algorithm_SVC_poly_kernel_configuration(metrics, 'SVC with polynomial kernel - default params', X, y, stratify = stratify, train_size = train_size, normalize_data = normalize_data)

    # CALIBRATING degree, gamma and coef0
    for degree in range(1, 20):
        run_algorithm_SVC_poly_kernel_configuration(metrics, 'SVC with polynomial kernel - degree = ' + str(degree), X, y, degree = degree, stratify = stratify, train_size = train_size, normalize_data = normalize_data)
        run_algorithm_SVC_poly_kernel_configuration(metrics, 'SVC with polynomial kernel - degree = ' + str(degree) + ', gamma = auto', X, y, degree = degree, gamma = 'auto', stratify = stratify, train_size = train_size, normalize_data = normalize_data)
        for i in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]:
            for gamma in [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01,
                          0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
                run_algorithm_SVC_poly_kernel_configuration(metrics, 'SVC with polynomial kernel - degree = ' + str(degree) + ', gamma = ' + str(gamma+ i), X, y, degree = degree, gamma =( gamma + i), stratify = stratify, train_size = train_size, normalize_data = normalize_data)
                for coef0 in [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 30, 40, 50, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]:
                    run_algorithm_SVC_poly_kernel_configuration(metrics, 'SVC with polynomial kernel - degree = ' + str(degree) + ', gamma = ' + str(gamma+ i) +', coef0 = ' + str(coef0), X, y,coef0 = coef0,  degree = degree, gamma =( gamma + i), stratify = stratify, train_size = train_size, normalize_data = normalize_data)



    # CALIBRATING tol
    for tol in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
        run_algorithm_SVC_poly_kernel_configuration(metrics, 'SVC with polynomial kernel - tol = '+ str(tol), X, y, tol = tol, stratify = stratify, train_size = train_size, normalize_data = normalize_data)
        run_algorithm_SVC_poly_kernel_configuration(metrics, 'SVC with polynomial kernel - shrinking = False, tol = '+ str(tol), X, y, tol = tol, shrinking=False,stratify = stratify, train_size = train_size, normalize_data = normalize_data)

    # CALIBRATING C
    for C in [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 100]:
        run_algorithm_SVC_poly_kernel_configuration(metrics, 'SVC with polynomial kernel - C = '+ str(C), X, y, C = C, stratify = stratify, train_size = train_size, normalize_data = normalize_data)
        run_algorithm_SVC_poly_kernel_configuration(metrics, 'SVC with polynomial kernel - shrinking = False, C = '+ str(C), X, y, C = C, shrinking=False,stratify = stratify, train_size = train_size, normalize_data = normalize_data)


    # CALIBRATING class_weight
    run_algorithm_SVC_poly_kernel_configuration(metrics, 'SVC with polynomial kernel - class_weight = balanced', X, y, class_weight='balanced', stratify = stratify, train_size = train_size, normalize_data = normalize_data)
    run_algorithm_SVC_poly_kernel_configuration(metrics, 'SVC with polynomial kernel - shrinking = False, class_weight = balanced', X, y, class_weight = 'balanced', shrinking=False,stratify = stratify, train_size = train_size, normalize_data = normalize_data)

    # CALIBRATING max_iter
    for max_iter in [100, 200, 300, 400, 500, 600, 800, 900, 925, 950, 975, 1000, 1025, 1050, 1075, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]:
        run_algorithm_SVC_poly_kernel_configuration(metrics, 'SVC with polynomial kernel - max_iter = '+ str(max_iter), X, y, max_iter = max_iter, stratify = stratify, train_size = train_size, normalize_data = normalize_data)
        run_algorithm_SVC_poly_kernel_configuration(metrics, 'SVC with polynomial kernel - shrinking = False, max_iter = '+ str(max_iter), X, y, shrinking = False, max_iter = max_iter, stratify = stratify, train_size = train_size, normalize_data = normalize_data)

    # CALIBRATING cache_size
    for cache_size in [100, 125, 150, 175, 190, 200, 210, 220, 230, 240, 250, 275, 300, 325, 350, 375, 400, 450, 500, 550, 600, 650, 700, 750, 800]:
        run_algorithm_SVC_poly_kernel_configuration(metrics, 'SVC with polynomial kernel - cache_size = '+ str(cache_size), X, y, cache_size = cache_size, stratify = stratify, train_size = train_size, normalize_data = normalize_data)
        run_algorithm_SVC_poly_kernel_configuration(metrics, 'SVC with polynomial kernel - shrinking = False, cache_size = '+ str(cache_size), X, y, shrinking = False, cache_size = cache_size, stratify = stratify, train_size = train_size, normalize_data = normalize_data)

    #TODO: our proposed POLY kernel SVM

    # export metrics to CSV FILE
    df_metrics = pd.DataFrame(metrics)
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    my_filename = os.path.join(path_to_script, filename)
    df_metrics.to_csv(my_filename, encoding='utf-8', index= True)




def write_error_to_log_file( e, log_file = 'logs.txt'):
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    my_filename = os.path.join(path_to_script, log_file)

    #   Open a file with access mode 'a'
    with open(my_filename, "a+") as file_object:
        file_object.write(f'{str(e)}\n\n')

def convert_string_to_datetime(string):
    data = None
    try:
        data = datetime.strptime(string, '%d/%m/%Y %H:%M')
    except:
        data = datetime.strptime(string, '%Y-%m-%dT%H:%M:%S.%f%z')
    return data


def convert_datetime_to_timestamp(dt):
    try:
        timestamp = int(round(dt.timestamp()))
        return timestamp
    except Exception:
        return 0