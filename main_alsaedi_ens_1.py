import enum
import random

from sklearn.ensemble import VotingClassifier, RandomForestClassifier

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, ComplementNB

from algs_pakhare import preprocess_y_df
from main_pakhare import binarize_label, balance_dataset, split_data_into_training_testing, \
    load_dataset_alsaedi, randomly_select
from utils.tokenizer import apply_TFIDF
from utils.utils import appendMetricsTOCSV, current_ms, compute_statistics
import pyswarms as ps
import pandas as pd

from utils.wrapper import WrapperForPAC, WrapperForLinearSVC
import os
os.environ['NUMEXPR_MAX_THREADS'] = '56'
os.environ['NUMEXPR_NUM_THREADS'] = '56'

class EnsembleStrategy(enum.Enum):
    voting_hard = 'voting_hard'
    voting_soft = 'voting_soft'
    weight_voting_hard = 'weight_voting_hard'
    weight_voting_soft = 'weight_voting_soft'


class PSOOptionsStrategy(enum.Enum):
    exp_decay = 'exp_decay'
    nonlin_mod = 'nonlin_mod'
    lin_variation = 'lin_variation'


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



def get_weights_with_PSO(tfidf_X_train, y_train, ensemble, ens_strategy, n_processes,
                         n_iter_pso, n_particles, w_strategy, c1_strategy, c2_strategy):
    X_train_train, X_validation, y_train_traing, y_validation = train_test_split(tfidf_X_train, y_train, test_size=0.25,
                                                                                 stratify=y_train, shuffle=True)

    # results obtained for Alsaedi dataset D2
    # options = {'c1': 0.7904, 'c2': 0.671, 'w': 0.747}

    options = {'c1': round(random.uniform(0.5, 0.99), 3), 'c2': round(random.uniform(0.5, 0.99), 3),
               'w': round(random.uniform(0.5, 0.99), 3)}

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


def  add_stats_to_results_ens(df_results, vectorizer, ensemble, strategy, cost, n_processes_pso, n_iter_pso,
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
    n_iter_pso = 100
    n_particles = 10

    sgd = SGDClassifier(class_weight='balanced', eta0=0.8102054986375052, learning_rate='constant',
                        loss='modified_huber', penalty=None)
    pac = WrapperForPAC(class_weight='balanced')
    linearSVC = WrapperForLinearSVC(class_weight='balanced')
    rf = RandomForestClassifier(class_weight='balanced', criterion='entropy', n_estimators=100)
    bnb = BernoulliNB(alpha=0.10460371036792848, binarize=0.042937660594707605, fit_prior=True)
    mnb = MultinomialNB(alpha=0.138110891026767)
    lr = LogisticRegression(class_weight='balanced', dual=False, max_iter=125, solver='saga')
    cnb = ComplementNB(alpha=0.41761996152417746, norm=True)

    for i in range(no_iterations):
        df = balance_dataset(df)
        df = randomly_select(no=20000, df=df)
        X_train_df, X_test_df, y_train_df, y_test_df = split_data_into_training_testing(df, testing_ratio=0.25)
        # w_strategy = PSOOptionsStrategy.exp_decay
        # c1_strategy = PSOOptionsStrategy.exp_decay
        # c2_strategy = PSOOptionsStrategy.lin_variation
        # ens_strategy = EnsembleStrategy.weight_voting_soft
        for w_strategy in PSOOptionsStrategy:
            for c1_strategy in PSOOptionsStrategy:
                for c2_strategy in PSOOptionsStrategy:
        # for n_iter_pso in [10, 100, 300, 500]:
        #     for n_particles in [10, 30, 50, 80, 100]:
                    top3_ensemble = [('sgd', sgd), ('pac', pac), ('linearSVC', linearSVC)]
                    # top5_ensemble = [('sgd', sgd), ('pac', pac), ('linearSVC', linearSVC), ('rf', rf), ('bnb', bnb)]
                    # top7_ensemble = [('sgd', sgd), ('pac', pac), ('linearSVC', linearSVC), ('rf', rf), ('bnb', bnb),
                    #              ('mnb', mnb), ('lr', lr)]
                    for ens_strategy in [EnsembleStrategy.weight_voting_soft, EnsembleStrategy.weight_voting_hard]:
                        df_results_ens = train_ensemble_with_PSO(df_results_ens, X_train_df, X_test_df, y_train_df, y_test_df,
                                                 # PSO configuration
                                                 w_strategy=w_strategy.value,
                                                 c1_strategy=c1_strategy.value,
                                                 c2_strategy=c2_strategy.value, n_iter_pso=n_iter_pso,
                                                 n_particles=n_particles, ensemble=top3_ensemble,
                                                 ens_strategy=ens_strategy,
                                                 n_processes_pso=n_processes_pso)
                        # df_results_ens = train_ensemble_with_PSO(df_results_ens, X_train_df, X_test_df, y_train_df, y_test_df,
                        #                          # PSO configuration
                        #                          w_strategy=w_strategy.value,
                        #                          c1_strategy=c1_strategy.value,
                        #                          c2_strategy=c2_strategy.value, n_iter_pso=n_iter_pso,
                        #                          n_particles=n_particles, ensemble=top7_ensemble,
                        #                          ens_strategy=ens_strategy,
                        #                          n_processes_pso=n_processes_pso)
                        # df_results_ens = appendMetricsTOCSV(filename, df_results_ens, init_function=init_results_df_ens)
                        # df_results_ens = train_ensemble_with_PSO(df_results_ens, X_train_df, X_test_df, y_train_df, y_test_df,
                        #                          # PSO configuration
                        #                          w_strategy=w_strategy.value,
                        #                          c1_strategy=c1_strategy.value,
                        #                          c2_strategy=c2_strategy.value, n_iter_pso=n_iter_pso,
                        #                          n_particles=n_particles, ensemble=top5_ensemble,
                        #                          ens_strategy=ens_strategy,
                        #                          n_processes_pso=n_processes_pso)
                        df_results_ens = appendMetricsTOCSV(filename, df_results_ens, init_function=init_results_df_ens)



if __name__ == "__main__":
    filename = 'out_ens_alsaedi_top3_faza1.csv'
    df = load_dataset_alsaedi()
    df = binarize_label(df)
    run_ensemble_with_PSO(df, filename, no_iterations=1, n_processes_pso=30)

