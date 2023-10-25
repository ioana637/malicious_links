from sklearn.linear_model import SGDClassifier

from algs_pakhare import add_stats_to_results
from main_pakhare import load_dataset_pakhare, binarize_label, balance_dataset, split_data_into_training_testing, \
    load_dataset_alsaedi, randomly_select
from utils.tokenizer import apply_TFIDF
from utils.utils import init_results_df, appendMetricsTOCSV, current_ms, preprocess_y_df, compute_statistics


def train_sgd(clf, df_results, filename, X_train_df, X_test_df, y_train_df, y_test_df):
    start_ms = current_ms()
    tfidf_X_train, tfidf_X_test = apply_TFIDF(X_train_df, X_test_df)

    classifier_name = str(clf.__class__.__name__)

    y_train, y_test = preprocess_y_df(y_train_df, y_test_df)
    clf.fit(tfidf_X_train, y_train)

    y_pred_train = clf.predict(tfidf_X_train)
    stats_training = compute_statistics(start_ms, y_pred_train, y_train)

    start_ms = current_ms()
    y_pred_test = clf.predict(tfidf_X_test)
    stats_testing = compute_statistics(start_ms, y_pred_test, y_test)

    df_results = add_stats_to_results(df_results, 'tfidf', classifier_name, stats_training,
                                      stats_testing, '')
    df_results = appendMetricsTOCSV(filename, df_results, init_function=init_results_df)
    return df_results

def run_sgd_for_pakhare(iterations=10):
    filename = 'out_ens_d1_sgd_best_est.csv'
    df_results = init_results_df()
    appendMetricsTOCSV(filename, df_results, init_function=init_results_df, header=True)
    df = load_dataset_pakhare()
    df = binarize_label(df)
    for i in range(iterations):
        df = balance_dataset(df)
        X_train_df, X_test_df, y_train_df, y_test_df = split_data_into_training_testing(df, testing_ratio=0.25)
        sgd = SGDClassifier(class_weight='balanced', eta0=0.2476923834458562, learning_rate='adaptive',
                        loss='modified_huber', penalty=None)
        df_results = train_sgd(sgd, df_results, filename, X_train_df, X_test_df, y_train_df, y_test_df)

def run_sgd_for_alsaedi(iterations=10):
    filename = 'out_ens_d2_sgd_best_est.csv'
    df_results = init_results_df()
    appendMetricsTOCSV(filename, df_results, init_function=init_results_df, header=True)
    df = load_dataset_alsaedi()
    df = binarize_label(df)
    for i in range(iterations):
        df = balance_dataset(df)
        df = randomly_select(no=20000, df=df)
        X_train_df, X_test_df, y_train_df, y_test_df = split_data_into_training_testing(df, testing_ratio=0.25)
        sgd = SGDClassifier(class_weight='balanced', eta0=0.8102054986375052, learning_rate='constant',
                            loss='modified_huber', penalty=None)
        df_results = train_sgd(sgd, df_results, filename, X_train_df, X_test_df, y_train_df, y_test_df)

if __name__ == "__main__":
    run_sgd_for_alsaedi(iterations=100)
    run_sgd_for_pakhare(iterations=100)