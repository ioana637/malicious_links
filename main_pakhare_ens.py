import enum
import random

from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, ComplementNB

from algs_pakhare import preprocess_y_df
from main_pakhare import load_dataset_pakhare, binarize_label, balance_dataset, split_data_into_training_testing
from utils.tokenizer import apply_TFIDF
from utils.utils import appendMetricsTOCSV, current_ms, compute_statistics
import pyswarms as ps
import pandas as pd

from utils.wrapper import WrapperForPAC, WrapperForLinearSVC


class EnsembleStrategy(enum.Enum):
    voting_hard = 'voting_hard',
    voting_soft = 'voting_soft',
    weight_voting_hard = 'weight_voting_hard',
    weight_voting_soft = 'weight_voting_soft'


class PSOOptionsStrategy(enum.Enum):
    exp_decay = 'exp_decay',
    nonlin_mod = 'nonlin_mod',
    lin_variation = 'lin_variation',


def init_results_df_ens():
    df_results = pd.DataFrame(columns=['ensemble_type', 'strategy', 'weights', 'estimators',
                                       'best_cost', 'c1_strategy', 'c2_strategy', 'w_strategy', 'n_inter_pso',
                                       'n_particles', 'n_processes_pso', 'c1', 'c2', 'w',
                                       'features', 'training_time', 'testing_time',

                                       'train_accuracy', 'train_precision_0', 'train_precision_1', 'train_recall_0',
                                       'train_recall_1', 'train_f1_0', 'train_f1_1', 'train_mcc',
                                       'train_support_0', 'train_support_1', 'train_support',
                                       'train_macro_avg_precision', 'train_macro_avg_recall', 'train_macro_avg_f1',
                                       'train_weighted_avg_precision', 'train_weighted_avg_recall',
                                       'train_weighted_avg_f1',

                                       'test_accuracy', 'test_precision_0', 'test_precision_1', 'test_recall_0',
                                       'test_recall_1', 'test_f1_0', 'test_f1_1', 'test_mcc',
                                       'test_support_0', 'test_support_1', 'test_support',
                                       'test_macro_avg_precision', 'test_macro_avg_recall', 'test_macro_avg_f1',
                                       'test_weighted_avg_precision', 'test_weighted_avg_recall',
                                       'test_weighted_avg_f1',
                                       ])

    return df_results


def get_percent_of_records(tfidf_X_train):
    return tfidf_X_train


def objective_function(x, ensemble, X_validation, y_validation, X_training, y_training, ens_strategy):
    acc_scores = []

    for x_particule in x:
        ens_clf = create_ensemble(x_particule, ens_strategy, ensemble)

        # print(ens_clf)
        ens_clf.fit(X_training, y_training)
        pred = ens_clf.predict(X_validation)
        acc_score = accuracy_score(y_validation, pred)
        # print(acc_score)
        acc_scores.append(1 - acc_score)

    # return accuracy scores in weights size for each particle
    return acc_scores


def sphere_minus_1(x):
    """Sphere objective function.

    Has a global minimum at :code:`0` and with a search domain of
        :code:`[-inf, inf]`

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`
    """
    j = 1 - ((x ** 2.0).sum(axis=1))
    # j = ((x ** 2.0).sum(axis=1))
    print(j)
    return j


def get_weights_with_PSO(tfidf_X_train, y_train, ensemble, ens_strategy, n_processes,
                         n_iter_pso, n_particles, w_strategy, c1_strategy, c2_strategy):
    X_train_train, X_validation, y_train_traing, y_validation = train_test_split(tfidf_X_train, y_train, test_size=0.25,
                                                                                 stratify=y_train, shuffle=True)

    # options = {'c1': 0.8, 'c2': 0.6, 'w': 0.9}
    options = {'c1': round(random.uniform(0.5, 0.99),3), 'c2': round(random.uniform(0.5, 0.99),3),
               'w': round(random.uniform(0.5, 0.99),3)}
    # C1 cognitive parameter
    # C2 social parameter
    # W inertia parameter

    # As done by others :)
    # optimizer = ps.single.GlobalBestPSO(n_particles=x7_valid.shape[0],dimensions=x7_valid.shape[1], options=options)
    # cost, pos = optimizer.optimize(fx.sphere, iters=1000)
    # oh_strategy = {"w": w_strategy[0], "c1": c1_strategy[0], "c2": c2_strategy[0]}
    oh_strategy = {"w": w_strategy, "c1": c1_strategy, "c2": c2_strategy}

    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles,
                                        dimensions=len(ensemble),
                                        options=options,
                                        oh_strategy=oh_strategy)

    # cost, pos = optimizer.optimize(sphere_minus_1, iters=50)
    cost, pos = optimizer.optimize(objective_function, iters=n_iter_pso, n_processes=n_processes,
                                   # arguments for the objective_function
                                   ensemble=ensemble, X_validation=X_validation, y_validation=y_validation,
                                   X_training=X_train_train, y_training=y_train_traing, ens_strategy=ens_strategy)
    weights = list(pos[0:len(ensemble)])
    return weights, cost, options['c1'], options['c2'], options['w']


def train_ensemble_with_PSO(df_results, X_train_df, X_test_df, y_train_df, y_test_df,
                            n_processes_pso: int,
                            w_strategy: PSOOptionsStrategy, c1_strategy: PSOOptionsStrategy,
                            c2_strategy: PSOOptionsStrategy,
                            n_iter_pso: int, n_particles: int, ensemble, ens_strategy: EnsembleStrategy):
    start_ms = current_ms()
    tfidf_X_train, tfidf_X_test = apply_TFIDF(X_train_df, X_test_df)
    y_train, y_test = preprocess_y_df(y_train_df, y_test_df)

    # WEIGHTS = np.random.dirichlet(np.ones(len(MY_ENSEMBLE)), size=1)
    weights, cost, c1, c2, w = get_weights_with_PSO(tfidf_X_train, y_train, ensemble, ens_strategy,
                                                    n_processes=n_processes_pso,
                                                    n_iter_pso=n_iter_pso, n_particles=n_particles,
                                                    w_strategy=w_strategy,
                                                    c1_strategy=c1_strategy, c2_strategy=c2_strategy)
    # Creating the ensemble
    ens_clf = create_ensemble(weights, ens_strategy, ensemble)

    # Training and testing the ensemble
    ens_clf.fit(tfidf_X_train, y_train)
    print('Training of %s completed in %d ms' % (ens_strategy, (current_ms() - start_ms)))
    ens_pred_train = ens_clf.predict(tfidf_X_train)
    stats_training = compute_statistics(start_ms, ens_pred_train, y_train)
    start_ms = current_ms()
    ens_pred_test = ens_clf.predict(tfidf_X_test)
    stats_testing = compute_statistics(start_ms, ens_pred_test, y_test)

    df_results = add_stats_to_results_ens(df_results, 'tfidf', ens_clf, ens_strategy, cost, n_processes_pso, n_iter_pso,
                                          n_particles, w_strategy, c1_strategy, c2_strategy, w, c1, c2,
                                          stats_training, stats_testing)
    return df_results


def create_ensemble(WEIGHTS, ens_strategy, ensemble):
    if ens_strategy == EnsembleStrategy.voting_hard:
        ens_clf = VotingClassifier(ensemble, voting='hard')
    elif ens_strategy == EnsembleStrategy.voting_soft:
        ens_clf = VotingClassifier(ensemble, voting='soft')
    elif ens_strategy == EnsembleStrategy.weight_voting_hard:
        ens_clf = VotingClassifier(ensemble, voting='hard', weights=WEIGHTS)
    else:
        ens_strategy = EnsembleStrategy.weight_voting_soft
        ens_clf = VotingClassifier(ensemble, voting='soft', weights=WEIGHTS)
    return ens_clf


def add_stats_to_results_ens(df_results, vectorizer, ensemble, strategy, cost, n_processes_pso, n_iter_pso,
                             n_particles, w_strategy, c1_strategy, c2_strategy, w, c1, c2,
                             stats_training, stats_testing):
    (training_time, train_report, mcc_train) = stats_training
    (testing_time, test_report, mcc_test) = stats_testing

    new_row = {'ensemble_type': str(ensemble.__class__.__name__), 'strategy': strategy,
               'weights': str(ensemble.weights),
               'estimators': str(ensemble.estimators_),
               'best_cost': cost, 'c1_strategy': c1_strategy, 'c2_strategy': c2_strategy, 'w_strategy': w_strategy,
               'n_inter_pso': n_iter_pso, 'n_particles': n_particles, 'n_processes_pso': n_processes_pso,
               'c1': c1, 'c2': c2, 'w': w,

               'features': vectorizer,
               'training_time': training_time,
               'testing_time': testing_time,

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


# pso:{'n_iter':int, 'n_particles':int, 'w_strategy': PSOOptionsStrategy, 'c1_strategy': PSOOptionsStrategy, 'c2_strategy': PSOOptionsStrategy}
def run_ensemble_with_PSO(df, filename, no_iterations: int, n_processes_pso):
    df_results_ens = init_results_df_ens()
    appendMetricsTOCSV(filename, df_results_ens, init_function=init_results_df_ens, header=True)
    n_iter_pso = 200
    n_particles = 20

    # sgd = SGDClassifier(class_weight='balanced', eta0=0.2476923834458562, learning_rate='adaptive', loss='hinge', penalty=None)
    sgd = SGDClassifier(class_weight='balanced', eta0=0.2476923834458562, learning_rate='adaptive',
                        loss='modified_huber', penalty=None)
    pac = WrapperForPAC(class_weight='balanced')
    linearSVC = WrapperForLinearSVC(class_weight='balanced', dual=True, fit_intercept=True, loss='squared_hinge',
                                    penalty='l2')
    # linearSVC = SVC(kernel='linear',probability=True, class_weight='balanced')
    bnb = BernoulliNB(alpha=0.05857603724009097, binarize=0.03756792108913587, fit_prior=False)
    cnb = ComplementNB(alpha=0.08533942891319435, norm=False)
    mnb = MultinomialNB(alpha=0.07759048342015018)
    lr = LogisticRegression(class_weight='balanced', dual=False, max_iter=100, solver='saga')

    for i in range(no_iterations):
        df = balance_dataset(df)
        X_train_df, X_test_df, y_train_df, y_test_df = split_data_into_training_testing(df, testing_ratio=0.25)
        for w_strategy in PSOOptionsStrategy:
            for c1_strategy in PSOOptionsStrategy:
                for c2_strategy in PSOOptionsStrategy:
                    top3_ensemble = [('sgd', sgd), ('pac', pac), ('linearSVC', linearSVC)]
                    top5_ensemble = [('sgd', sgd), ('pac', pac), ('linearSVC', linearSVC), ('bnb', bnb), ('cnb', cnb)]
                    top7_ensemble = [('sgd', sgd), ('pac', pac), ('linearSVC', linearSVC), ('bnb', bnb), ('cnb', cnb),
                                     ('mnb', mnb), ('lr', lr)]
                    for ens_strategy in [EnsembleStrategy.weight_voting_soft, EnsembleStrategy.weight_voting_hard]:
                        df_results_ens = train_ensemble_with_PSO(df_results_ens, X_train_df, X_test_df, y_train_df, y_test_df,
                                                # PSO configuration
                                                w_strategy=w_strategy.value[0], c1_strategy=c1_strategy.value[0],
                                                c2_strategy=c2_strategy.value[0], n_iter_pso=n_iter_pso,
                                                n_particles=n_particles, ensemble=top3_ensemble,
                                                ens_strategy=ens_strategy.value, n_processes_pso=n_processes_pso)
                        df_results_ens = train_ensemble_with_PSO(df_results_ens, X_train_df, X_test_df, y_train_df, y_test_df,
                                                # PSO configuration
                                                w_strategy=w_strategy.value[0], c1_strategy=c1_strategy.value[0],
                                                c2_strategy=c2_strategy.value[0], n_iter_pso=n_iter_pso,
                                                n_particles=n_particles, ensemble=top7_ensemble,
                                                ens_strategy=ens_strategy.value, n_processes_pso=n_processes_pso)
                        df_results_ens = train_ensemble_with_PSO(df_results_ens, X_train_df, X_test_df, y_train_df, y_test_df,
                                                # PSO configuration
                                                w_strategy=w_strategy.value[0], c1_strategy=c1_strategy.value[0],
                                                c2_strategy=c2_strategy.value[0], n_iter_pso=n_iter_pso,
                                                n_particles=n_particles, ensemble=top5_ensemble,
                                                ens_strategy=ens_strategy.value, n_processes_pso=n_processes_pso)
                    appendMetricsTOCSV(filename, df_results_ens, init_function=init_results_df_ens)
        # df_results_ens = train_ensemble_with_PSO(df_results_ens, X_train_df, X_test_df, y_train_df, y_test_df,
        #                                          # PSO configuration
        #                                          w_strategy=PSOOptionsStrategy.nonlin_mod.value[0],
        #                                          c1_strategy=PSOOptionsStrategy.nonlin_mod.value[0],
        #                                          c2_strategy=PSOOptionsStrategy.nonlin_mod.value[0], n_iter_pso=n_iter_pso,
        #                                          n_particles=n_particles, ensemble=top3_ensemble,
        #                                          ens_strategy=EnsembleStrategy.weight_voting_soft,
        #                                          n_processes_pso=n_processes_pso)
        # df_results_ens = train_ensemble_with_PSO(df_results_ens, X_train_df, X_test_df, y_train_df, y_test_df,
        #                                          # PSO configuration
        #                                          w_strategy=PSOOptionsStrategy.nonlin_mod.value[0],
        #                                          c1_strategy=PSOOptionsStrategy.nonlin_mod.value[0],
        #                                          c2_strategy=PSOOptionsStrategy.nonlin_mod.value[0], n_iter_pso=n_iter_pso,
        #                                          n_particles=n_particles, ensemble=top7_ensemble,
        #                                          ens_strategy=EnsembleStrategy.weight_voting_soft,
        #                                          n_processes_pso=n_processes_pso)
        # df_results_ens = train_ensemble_with_PSO(df_results_ens, X_train_df, X_test_df, y_train_df, y_test_df,
        #                                          # PSO configuration
        #                                          w_strategy=PSOOptionsStrategy.nonlin_mod.value[0],
        #                                          c1_strategy=PSOOptionsStrategy.nonlin_mod.value[0],
        #                                          c2_strategy=PSOOptionsStrategy.nonlin_mod.value[0], n_iter_pso=n_iter_pso,
        #                                          n_particles=n_particles, ensemble=top5_ensemble,
        #                                          ens_strategy=EnsembleStrategy.weight_voting_soft,
        #                                          n_processes_pso=n_processes_pso)



if __name__ == "__main__":
    filename = 'out_ens_2.csv'
    df = load_dataset_pakhare()
    df = binarize_label(df)
    run_ensemble_with_PSO(df, filename, no_iterations=5, n_processes_pso=60)