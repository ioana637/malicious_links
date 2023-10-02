import numpy as np

# Functions to compute diversity between algorithms
def compute_n(pred1: np.ndarray, pred2: np.ndarray, test_y: np.ndarray):
    """
    Method to compute n00, n01, n10, n11
    :param pred1: predictions of the first classifier
    :param pred2: predictions of the second classifier
    :param test_y: reference test labels
    :return: n00, n01, n10, n11
    """
    n11 = sum((pred1 == test_y) * (pred2 == test_y))
    n10 = sum((pred1 == test_y) * (pred2 != test_y))
    n01 = sum((pred1 != test_y) * (pred2 == test_y))
    n00 = sum((pred1 != test_y) * (pred2 != test_y))
    return n00, n01, n10, n11

def compute_qstat(pred1: np.ndarray, pred2: np.ndarray, test_y: np.ndarray) -> float:
    """
    Method to compute QStat between two arrays of predictions
    :param pred1: predictions of the first classifier
    :param pred2: predictions of the second classifier
    :param test_y: reference test labels
    :return: Q
    """
    n00, n01, n10, n11 = compute_n(pred1, pred2, test_y)
    return (n11*n00 - n01*n10)/(n11*n00 + n01*n10)

def compute_doublefault(pred1: np.ndarray, pred2: np.ndarray, test_y: np.ndarray) -> float:
    """
    Method to compute DoubleFault DF between two arrays of predictions
    :param pred1: predictions of the first classifier
    :param pred2: predictions of the second classifier
    :param test_y: reference test labels
    :return: DF
    """
    n00, n01, n10, n11 = compute_n(pred1, pred2, test_y)
    return n00/(n00 + n01 + n10 + n11)

def compute_ensemble_qstat(clf_predictions: dict, test_y: np.ndarray) -> float:
    """
    Abstract method to compute diversity metric
    :param clf_predictions: predictions of clfs in the ensemble
    :param test_y: reference test labels
    :return: a float value
    """
    clfs = list(clf_predictions.keys())
    n_clfs = len(clfs)
    diversities = [compute_qstat(clf_predictions[clfs[i]], clf_predictions[clfs[k]], test_y)
                   for i in range(0, n_clfs-1) for k in range(i+1, n_clfs)]
    return 2*sum(diversities)/(n_clfs*(n_clfs-1))

def diversity_report(predictions_dict: dict, y_true: np.ndarray, detailed = False):
    if predictions_dict is not None and len(predictions_dict) > 1:
        if detailed == True:
            print('\nPair-wise diversity:')
            clfs = list(predictions_dict.keys())
            for i in range(0, len(clfs)):
                for k in range(i+1, len(clfs)):
                    qstat = compute_qstat(predictions_dict[clfs[i]], predictions_dict[clfs[k]], y_true)
                    print(' QStat(%s, %s): %.3f' % (clfs[i], clfs[k], qstat))
                    df = compute_doublefault(predictions_dict[clfs[i]], predictions_dict[clfs[k]], y_true)
                    print(' DF(%s, %s): %.3f' % (clfs[i], clfs[k], df))
        ens_q = compute_ensemble_qstat(predictions_dict, y_true)
        print('Q-Stat Ensemble diversity: %.3f\n' % ens_q)

