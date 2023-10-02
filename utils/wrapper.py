from sklearn.linear_model import PassiveAggressiveClassifier
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC


class WrapperForLinearSVC(LinearSVC):
    def __init__(self, penalty='l2', loss='squared_hinge', *, dual=True,
                 tol=1e-4, C=1.0, multi_class='ovr', fit_intercept=True,
                 intercept_scaling=1, class_weight=None, verbose=0,
                 random_state=None, max_iter=1000):
        super().__init__(penalty = penalty, loss = loss, dual = dual, tol= tol, C=C, multi_class=multi_class,
                         fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight,
                         verbose=verbose, random_state = random_state, max_iter = max_iter)

    def predict_proba(self, x_test) -> np.ndarray:
        """
        Method to compute probabilities of predicted classes
        :return: array of probabilities for each classes
        """
        return self._predict_proba_lr(x_test)


class WrapperForPAC(PassiveAggressiveClassifier):
    def __init__(self, *, C=1.0, fit_intercept=True, max_iter=1000, tol=1e-3,
                 early_stopping=False, validation_fraction=0.1,
                 n_iter_no_change=5, shuffle=True, verbose=0, loss="hinge",
                 n_jobs=None, random_state=None, warm_start=False,
                 class_weight=None, average=False):
        super().__init__(
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            shuffle=shuffle,
            verbose=verbose,
            random_state=random_state,
            warm_start=warm_start,
            class_weight=class_weight,
            average=average,
            n_jobs=n_jobs, C = C, loss = loss)

    def predict_proba(self, x_test) -> np.ndarray:
        """
        Method to compute probabilities of predicted classes
        :return: array of probabilities for each classes
        """
        return self._predict_proba_lr(x_test)



