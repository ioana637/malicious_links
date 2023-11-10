import enum
import time
from itertools import product

import numpy as np
import pandas as pd
from pyod.models.copod import COPOD
from pyod.models.iforest import IForest
from scipy.stats import uniform
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.covariance import EmpiricalCovariance, EllipticEnvelope, GraphicalLassoCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, Perceptron, ElasticNet, SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from utils.dense_transformer import DenseTransformer
from utils.tokenizer import tokenizer, tokenizer_pakhare, apply_TFIDF
from utils.utils import appendMetricsTOCSV, init_results_df, compute_statistics, current_ms, preprocess_y_df


class Algorithms(enum.Enum):
    RF = 'RandomForestClassifier',
    SVC = 'SVC',
    ada = 'AdaBoostClassifier',
    histXGB = 'HistGradientBoostingClassifier',
    bnb = 'BernoulliNB',
    mnb = 'MultinomialNB',
    gnb = 'GaussianNB',
    cnb = 'ComplementNB',
    lr = 'LogisticRegression',
    copod = 'COPOD',
    iforest = 'IForest',
    lda = 'LinearDiscriminantAnalysis',
    knn = 'KNeighborsClassifier',
    xgb = 'GradientBoostingClassifier',
    linearSVC = 'LinearSVC',
    pac = 'PassiveAggressiveClassifier',
    perceptron = 'Perceptron',
    mlp = 'MLPClassifier',
    sgd = 'SGDClassifier',
    kmeans = 'KMeans',
    nc = 'NearestCentroid',
    dt = 'DecisionTreeClassifier'





parameters_rf = {'criterion': ['gini', 'entropy'], 'class_weight': ['balanced'],
                  'n_estimators': [100, 75, 125, 150]} #48
parameters_ada = {'n_estimators': [100, 75, 50, 125, 150], 'algorithm': ['SAMME', 'SAMME.R']} #10
parameters_knn = {'weights': ['distance'], 'n_neighbors': [3, 5]} #2

parameters_lr_all = list(product( [True, False], ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']))  #8
parameters_lr_filtered = [{'class_weight': ['balanced'], 'max_iter': [100, 75, 125, 150], 'solver': [solver],
                                  'dual' : [dual]} for dual, solver in parameters_lr_all
                                 if not (solver == 'newton-cg' and dual is True)
                                 and not (solver == 'lbfgs' and dual is True)
                                 and not (solver == 'saga' and dual is True)
                                 and not (solver == 'sag' and dual is True)]

# parameters_lr = {'class_weight': ['balanced'], 'max_iter': [100, 75, 125, 150],
#                  'solver': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'],
#                  'dual': [True, False]} #288
parameters_xgb = {'loss': ['exponential'], 'n_estimators': [75, 50, 100, 125, 150],
                  'criterion': ['friedman_mse'], 'max_leaf_nodes': [31, 48, 24, None],
                  'max_depth': [3, 8, 12, 24]
                  } # 80
parameters_hist_xgb = {'max_iter': [100, 75, 125, 150], 'interaction_cst': ('pairwise', 'no_interactions'),
                       'class_weight': ['balanced'], 'max_leaf_nodes': [31, 16, 48, 24, None],
                       'max_depth': [12, 16, 8, 24, 32, 36]}
parameters_lda = {'solver': ['svd', 'lsqr', 'eigen'], 'shrinkage': ('auto', None), 'store_covariance': (True, False),
                  'covariance_estimator': [EmpiricalCovariance(), EmpiricalCovariance(assume_centered=True),
                                           EllipticEnvelope(random_state=0), GraphicalLassoCV(),
                                           EllipticEnvelope(random_state=0, assume_centered=True)]}
parameters_svc = {'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
                  'degree': [3, 5, 7, 9],
                  # 'gamma': ('scale', 'auto'), 'shrinking': (True, False),
                  'probability': [True], 'class_weight': ['balanced']} # 20
parameters_iforest = {'contamination': [0.1, 0.2, 0.3, 0.4, 0.5], 'n_estimators': [100, 75, 50, 125, 150],
                      'bootstrap': (True, False)}
parameters_copod = {'contamination': [0.1, 0.2, 0.3, 0.4, 0.5]}

parameters_linearSVC_all = list(product( [True, False], ['l1', 'l2'], ['hinge', 'squared_hinge']))  #8

parameters_linearSVC_filtered = [{'class_weight': ['balanced'],'fit_intercept': [True, False],  'dual': [dual], 'penalty' : [penalty], 'loss': [loss]}
                   for dual, penalty, loss in parameters_linearSVC_all
                   if not (penalty == 'l1' and loss == 'hinge')
                   and not ((penalty == 'l1' and loss == 'squared_hinge' and dual is True))
                   and not ((penalty == 'l2' and loss == 'hinge' and dual is False))]

parameters_cnb = {'norm': [True, False], 'alpha': uniform()} #Random
parameters_mnb = {'alpha': uniform()} #Random
parameters_bnb = {'fit_prior': [True, False], 'alpha': uniform(), 'binarize': uniform() } #4

parameters_pac= {
    'class_weight': ['balanced'],
    # 'metric': ['nan_euclidean', 'manhattan', 'l1', 'l2', 'haversine', 'euclidean', 'cosine', 'cityblock']
}
# dict_keys(['C', 'average', 'class_weight', 'early_stopping', 'fit_intercept', 'loss', 'max_iter', 'n_iter_no_change', 'n_jobs', 'random_state', 'shuffle', 'tol', 'validation_fraction', 'verbose', 'warm_start'])

parameters_nc = {
    'metric': ['nan_euclidean', 'manhattan', 'l1', 'l2', 'haversine', 'euclidean', 'cosine', 'cityblock']
}
parameters_perceptron = {'penalty': ['l2', 'l1', 'elasticnet'], 'class_weight': ['balanced'], }
#TODO
parameters_mlp = {
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
}
# hidden_layer_sizes = [35, 100, 100]
parameters_sgd = {
    'class_weight': ['balanced'],
    'penalty':['l2', 'l1', 'elasticnet', None],
    'loss':['hinge', 'modified_huber', 'squared_hinge', 'perceptron', 'huber',
            'epsilon_insensitive', 'squared_epsilon_insensitive'],
    'learning_rate': ['optimal', 'constant', 'invscaling', 'adaptive'],
    'eta0': uniform()
} #128

parameters_kmeans = {
    'n_clusters': [2],
    'algorithm':[ 'elkan', 'auto', 'full']
}

parameters_dt = {
    'max_leaf_nodes': [31, 48, 24, None],
    'max_depth': [None, 12, 16, 8],
    'class_weight': ['balanced'],
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random']
}


distributions = {
    'DecisionTreeClassifier': {
        'params': parameters_dt,
        'search': 'grid'
    },
    'NearestCentroid':  {
        'params': parameters_nc,
        'search': 'grid'
    },
    'KMeans': {
        'params': parameters_kmeans,
        'search': 'grid'
    },
    'SGDClassifier': {
        'params': parameters_sgd,
        'search': 'random'
    },
    'MLPClassifier': {
        'params': parameters_mlp,
        'search': 'grid'
    },
    'Perceptron': {
        'params': parameters_perceptron,
        'search': 'grid'
    },
    'PassiveAggressiveClassifier': {
        'params': parameters_pac,
        'search': 'grid'
    },
    'LinearSVC': {
        'params': parameters_linearSVC_filtered,
        'search': 'grid'
    },
    'GradientBoostingClassifier': {
        'params': parameters_xgb,
        'search': 'grid'
    },
    'KNeighborsClassifier': {
        'params': parameters_knn,
        'search': 'grid'
    },
    'LinearDiscriminantAnalysis': {
        'params': parameters_lda,
        'search': 'grid'
    },
    'IForest': {
        'params': parameters_iforest,
        'search': 'grid'
    },
    'COPOD': {
        'params': parameters_copod,
        'search': 'grid'
    },
    'LogisticRegression': {
        'params': parameters_lr_filtered,
        'search': 'grid'
    },
    'ComplementNB': {
        'params': parameters_cnb,
        'search': 'random'
    },
    'BernoulliNB': {
        'params': parameters_bnb,
        'search': 'random'
    },
    'MultinomialNB': {
        'params': parameters_mnb,
        'search': 'random'
    },
    'HistGradientBoostingClassifier': {
        'params': parameters_hist_xgb,
        'search': 'grid'
    },
    'AdaBoostClassifier': {
        'params': parameters_ada,
        'search': 'grid'
    },
    'SVC': {
        'params': parameters_svc,
        'search': 'grid'
    },
    'RandomForestClassifier':{
        'params': parameters_rf,
        'search': 'grid'
    }
}


class PYODWrapper(BaseEstimator):
    """
    Abstract Class for PYOD algorithms
    Wraps implementation from different frameworks SKLEARN, PyOD etc.
    """

    def __init__(self, model):
        self.model = model
        self.trained = False
        self.estimator_type = 'classifier'
        self.classes = None
        self.feature_importance = None
        self.X = None
        self.y = None
        self.name = model.__class__.__name__
        self.revert = False

    def fit(self, x_train, y_train=None):
        if isinstance(x_train, pd.DataFrame):
            self.model.fit(x_train.to_numpy())
        else:
            self.model.fit(x_train)
        self.classes = 2
        self.feature_importance = self.compute_feature_importances()
        self.trained = True
        y_pred = self.predict(x_train)
        train_acc = accuracy_score(y_train, y_pred)
        if (train_acc < 0.5):
            self.revert = True
        train_prec = precision_score(y_train, y_pred)
        train_recall = recall_score(y_train, y_pred)
        train_f1 = f1_score(y_train, y_pred)

    def is_trained(self):
        return self.trained

    def predict(self, x_test):
        if isinstance(x_test, pd.DataFrame):
            x_t = x_test.to_numpy()
        else:
            x_t = x_test
        y_pred = self.model.predict(x_t)
        if self.revert:
            y_pred = abs(1 - y_pred)
        return y_pred

    def predict_proba(self, x_test) -> np.ndarray:
        """
        Method to compute probabilities of predicted classes
        :return: array of probabilities for each classes
        """
        test_features = x_test.to_numpy() if isinstance(x_test, pd.DataFrame) else x_test
        proba = self.model.predict_proba(test_features)
        pred = self.model.predict(test_features)
        for i in range(len(pred)):
            min_p = min(proba[i])
            max_p = max(proba[i])
            proba[i][pred[i]] = max_p
            proba[i][1 - pred[i]] = min_p
        if self.revert:
            proba[:] = proba[:, [1, 0]]
        return proba

    def compute_scores(self, proba) -> np.ndarray:
        """
        Method to compute accuracy scores of predicted classes
        F1 score, recall, accuracy, precision
        :return: array of scores for each classes
        """
        print(proba)
        return proba

    def predict_confidence(self, x_test: np.ndarray):
        """
        Method to compute confidence in the predicted class
        :return: -1 as default, value if algorithm is from framework PYOD
        """
        return -1

    def compute_feature_importances(self) -> np.ndarray:
        """
        Outputs feature ranking in building a Classifier
        :return: ndarray containing feature ranks
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.sum(np.absolute(self.model.coef_), axis=0)
        return []


def pipeline_for_algorithm_with_search(classifier, df_results, filename, X_train, X_test, y_train, y_test):
    start_ms = current_ms()
    tfidf_X_train, tfidf_X_test = apply_TFIDF(X_train, X_test)

    classifier_name = str(classifier.__class__.__name__)
    dist_clf = distributions[classifier_name]
    if (dist_clf.get('search') == 'grid'):
        # use Grid Search CV
        clf = GridSearchCV(classifier, dist_clf.get('params'), cv = 5, return_train_score = True, n_jobs = -1,
                           error_score=0.0)
    else:
    #     use Randomized SearchCV
        clf = RandomizedSearchCV(classifier,dist_clf.get('params'), cv = 5, return_train_score = True, n_jobs = -1,
                                 error_score=0.0)

    y_train, y_test = preprocess_y_df(y_train, y_test)
    clf.fit(tfidf_X_train, y_train)
    print(clf.best_params_)
    print(clf.best_estimator_)
    print(clf.best_score_)
    print(clf.best_index_)

    y_pred_train = clf.predict(tfidf_X_train)
    stats_training = compute_statistics(start_ms, y_pred_train, y_train)

    start_ms = current_ms()
    y_pred_test = clf.predict(tfidf_X_test)
    stats_testing = compute_statistics(start_ms, y_pred_test, y_test)

    df_results = add_stats_to_results(df_results, 'tfidf', classifier_name, stats_training,
                                      stats_testing, str(clf.best_params_))
    appendMetricsTOCSV(filename, df_results, init_function=init_results_df)



def pipeline_for_algorithm(classifier, df_results, X_train, X_test, y_train, y_test):
    start_ms = current_ms()
    tVec = TfidfVectorizer(tokenizer=tokenizer, stop_words='english', dtype=np.float32, ngram_range=(1, 1))
    tfidf_X_train = tVec.fit_transform(X_train['URLs'])
    tfidf_X_test = tVec.transform(X_test['URLs'])

    y_train, y_test = preprocess_y_df(y_train, y_test)
    classifier.fit(tfidf_X_train, y_train)
    y_pred_train = classifier.predict(tfidf_X_train)
    stats_training = compute_statistics(start_ms, y_pred_train, y_train)

    start_ms = current_ms()
    y_pred_test = classifier.predict(tfidf_X_test)
    stats_testing = compute_statistics(start_ms, y_pred_test, y_test)
    df_results = add_stats_to_results(df_results, 'tfidf', str(classifier.__class__.__name__), stats_training, stats_testing, '')


def add_stats_to_results(df_results, vectorizer, algorithm_name, stats_training, stats_testing, best_estimator):
    (training_time, train_report, mcc_train) = stats_training
    (testing_time, test_report, mcc_test) = stats_testing

    new_row = {'algorithm': algorithm_name, 'features': vectorizer, 'training_time': training_time,
               'testing_time': testing_time, 'best_estimator': best_estimator or '',
               'train_mcc': mcc_train, 'test_mcc': mcc_test,
               'train_accuracy': train_report['accuracy'], 'test_accuracy': test_report['accuracy'],
               'train_precision_0': train_report['0']['precision'],
               'train_precision_1': train_report['1']['precision'],
               'test_precision_0': test_report['0']['precision'],
               'test_precision_1': test_report['1']['precision'],
               'train_recall_0': train_report['0']['recall'],
               'train_recall_1': train_report['1']['recall'],
               'test_recall_0': test_report['0']['recall'],
               'test_recall_1': test_report['1']['recall'],
               'train_f1_0': train_report['0']['f1-score'],
               'train_f1_1': train_report['1']['f1-score'],
               'test_f1_0': test_report['0']['f1-score'],
               'test_f1_1': test_report['1']['f1-score'],
               'test_support_0': test_report['0']['support'],
               'test_support_1': test_report['1']['support'],
               'train_support_0': train_report['0']['support'],
               'train_support_1': train_report['1']['support'],
               'test_support': test_report['macro avg']['support'],
               'train_support': train_report['macro avg']['support'],

               'test_macro_avg_precision': test_report['macro avg']['precision'],
               'test_macro_avg_recall': test_report['macro avg']['recall'],
               'test_macro_avg_f1': test_report['macro avg']['f1-score'],
               'test_weighted_avg_precision': test_report['weighted avg']['precision'],
               'test_weighted_avg_recall': test_report['weighted avg']['recall'],
               'test_weighted_avg_f1': test_report['weighted avg']['f1-score'],

               'train_macro_avg_precision': train_report['macro avg']['precision'],
               'train_macro_avg_recall': train_report['macro avg']['recall'],
               'train_macro_avg_f1': train_report['macro avg']['f1-score'],
               'train_weighted_avg_precision': train_report['weighted avg']['precision'],
               'train_weighted_avg_recall': train_report['weighted avg']['recall'],
               'train_weighted_avg_f1': train_report['weighted avg']['f1-score'],
               }

    df_results = df_results.append(new_row, ignore_index=True)
    print(df_results)
    return df_results





# NOT uSED:
#     # ----------------------------DENSE DATA --------------------
#     # clustering
#     # TODO contamination with (y_train == 1).sum()/len(y_train)
#     copod = PYODWrapper(COPOD(n_jobs=-1))
#     iforest = PYODWrapper(IForest(n_jobs=-1))
#
#     hist_xgb = HistGradientBoostingClassifier()  # more suitable for larger datasets no_samples>10,000
#
#     lda = LinearDiscriminantAnalysis()
#     qda = QuadraticDiscriminantAnalysis()
#     gaussianNB = GaussianNB()
#     gpc = GaussianProcessClassifier(n_jobs=-1)
#     # ----------------------------DENSE DATA ------------------------------
#
#     #  Needs dense data: LDA, QDA, hist XGB, copod, iforest, gaussianNB, gpc

def train_individual_models(df_results, filename, X_train, X_test, y_train, y_test):
    # supervised
    multinomialNB = MultinomialNB()
    bernoulliNB = BernoulliNB()
    complementNB = ComplementNB()
    lr = LogisticRegression()
    dt = DecisionTreeClassifier()
    linearSVC = LinearSVC()
    pac = PassiveAggressiveClassifier(n_jobs=-1)
    perceptron = Perceptron(n_jobs=-1)
    sgd = SGDClassifier(n_jobs=-1)
    kmeans = KMeans(n_clusters=2)
    nc = NearestCentroid()
    # ensemble
    ada = AdaBoostClassifier()
    rf = RandomForestClassifier(n_jobs=-1)
    xgb = GradientBoostingClassifier()

    # SLOW
    knn = KNeighborsClassifier(n_jobs=-1)
    mlp = MLPClassifier()
    svc = SVC()

    classifiers = [
        # bernoulliNB , multinomialNB, complementNB, perceptron, pac, sgd, nc, lr, ada, dt, linearSVC,
        rf  # - on the server
         # svc, xgb, mlp - astea dureaza mult ...
    ]
    for cls in classifiers:
        print('Running '+str(cls.__class__.__name__))
        print(cls.get_params().keys())
        # pipeline_for_algorithm(cls, df_results, X_train, X_test, y_train, y_test)
        pipeline_for_algorithm_with_search(cls, df_results, filename, X_train, X_test, y_train, y_test)

