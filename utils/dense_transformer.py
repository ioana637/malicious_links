import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import TransformerMixin


class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        X = csr_matrix(X, dtype=np.int8)
        return X.todense()